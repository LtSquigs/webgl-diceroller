/**
 * dice_engine_lifecycle.cpp — DicePhysicsEngine construction, per-die setters,
 * and static-collider registration.
 */

#include "../dice_physics_engine.hpp"

#include <algorithm>
#include <cmath>

namespace dice_physics {

namespace {

void poseStaticHull(StaticBody& s) {
    s.worldVerts.resize(s.hull.verts.size());
    transformHullVerts(s.hull.verts.data(), static_cast<int>(s.hull.verts.size()), s.rotation,
                       s.center, s.worldVerts.data());
}

} // namespace

DicePhysicsEngine::DicePhysicsEngine()
    : gravity_(-15.0f), tableY_(-2.75f), tableHalfW_(18.0f), tableHalfD_(18.0f), nextId_(0) {
    // Collision events accumulate until the host drains them, capped at
    // MAX_EVENTS_PER_STEP; reserving the cap keeps step() off the heap.
    events_.reserve(MAX_EVENTS_PER_STEP);
    // Two posed planes per face of the largest hull addSatContacts can meet.
    satPlanesA_.reserve(2 * static_cast<size_t>(MAX_VERTICES_PER_HULL));
    satPlanesB_.reserve(2 * static_cast<size_t>(MAX_VERTICES_PER_HULL));
}

void DicePhysicsEngine::reserveContactStorage() {
    const size_t want =
        std::min(MAX_MANIFOLDS, 8 * (bodies_.size() + dynamics_.size()) + 32);
    if (manifolds_.capacity() < want) {
        manifolds_.reserve(std::min(MAX_MANIFOLDS, std::max(want, 2 * manifolds_.capacity())));
    }
    const size_t cap = manifolds_.capacity();
    if (solveOrder_.capacity() < cap) solveOrder_.reserve(cap);
    if (manifoldBodySlots_.capacity() < cap) manifoldBodySlots_.reserve(cap);
    if (islandItems_.capacity() < cap) islandItems_.reserve(cap);
}

void DicePhysicsEngine::setFlags(uint32_t flags) {
    noDrag_ = (flags & FLAG_NO_DRAG) != 0;
    fairDice_ = (flags & FLAG_FAIR_DICE) != 0;
}

void DicePhysicsEngine::setMassBiasRatio(float ratio) {
    massBiasRatio_ = std::isfinite(ratio) ? std::clamp(ratio, 0.0f, MAX_MASS_BIAS_RATIO)
                                          : DEFAULT_MASS_BIAS_RATIO;
}

void DicePhysicsEngine::init(float gravity, float tableY, float tableHalfW, float tableHalfD) {
    gravity_ = gravity; tableY_ = tableY;
    tableHalfW_ = tableHalfW; tableHalfD_ = tableHalfD;
    bodies_.clear(); manifolds_.clear(); events_.clear();
    statics_.clear();
    dynamics_.clear();
    nextId_ = 0;
    gridCols_ = 0;
    gridRows_ = 0;
    lastStepStats_ = {};
    accumulator_ = 0.0;
    staticCapacityDroppedCount_ = 0;
    dynamicCapacityDroppedCount_ = 0;
}

void DicePhysicsEngine::reset() {
    bodies_.clear(); manifolds_.clear(); events_.clear(); statics_.clear(); dynamics_.clear();
    nextId_ = 0;
    accumulator_ = 0.0;
    staticCapacityDroppedCount_ = 0;
    dynamicCapacityDroppedCount_ = 0;
}

int DicePhysicsEngine::addDie(int sides, float x, float y, float z) {
    if (bodies_.size() >= static_cast<size_t>(MAX_DICE)) return -1;
    if (!allFinite(x, y, z)) return -1;
    RigidBody b;
    b.id = nextId_++;
    b.sides = sides;
    b.position = {x, y, z};
    b.rotation = {0,0,0,1};
    b.radius = radiusForSides(sides);
    b.mass = 5.0f;
    b.invMass = 1.0f / b.mass;
    b.computeInertiaFromHull();
    bodies_.push_back(b);
    reserveContactStorage();
    return b.id;
}

void DicePhysicsEngine::removeDie(int id) {
    bodies_.erase(
        std::remove_if(bodies_.begin(), bodies_.end(),
            [id](const RigidBody& b) { return b.id == id; }),
        bodies_.end());
}

void DicePhysicsEngine::clearAllDice() { bodies_.clear(); manifolds_.clear(); events_.clear(); }

void DicePhysicsEngine::setDieMaterial(int id, float friction, float rollingFriction) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.friction = std::clamp(friction, 0.0f, 2.0f);
        b.rollingFriction = std::clamp(rollingFriction, 0.0f, 1.0f);
        break;
    }
}

void DicePhysicsEngine::setDieDrag(int id, float dragFactor) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.dragFactor = std::max(0.0f, dragFactor);
        break;
    }
}

void DicePhysicsEngine::setDieHull(int id, const std::vector<float>& flatVerts) {
    if (flatVerts.size() % 3 != 0) return;
    if (flatVerts.size() / 3 > MAX_VERTICES_PER_HULL) return;
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        std::vector<Vec3> verts;
        verts.reserve(flatVerts.size() / 3);
        for (size_t i = 0; i < flatVerts.size(); i += 3) {
            float vx = flatVerts[i], vy = flatVerts[i+1], vz = flatVerts[i+2];
            if (!allFinite(vx, vy, vz)) continue;
            verts.push_back({vx, vy, vz});
        }
        b.hull.build(verts);
        b.useHull = true;
        b.computeInertiaFromHull();
        b.refreshComAxis();
        break;
    }
}

void DicePhysicsEngine::applyImpulse(int id, float fx, float fy, float fz) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.velocity += Vec3{fx, fy, fz} * b.invMass;
        wake(b);
        break;
    }
}

void DicePhysicsEngine::applyTorqueImpulse(int id, float tx, float ty, float tz) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.angularVelocity += b.applyInvInertiaWorld(Vec3{tx, ty, tz});
        wake(b);
        break;
    }
}

void DicePhysicsEngine::setDieTransform(int id, float px, float py, float pz,
                     float qx, float qy, float qz, float qw) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.position = {px, py, pz};
        b.rotation = Quat{qx, qy, qz, qw}.normalized();
        b.velocity = {};
        b.angularVelocity = {};
        wake(b);
        break;
    }
}

void DicePhysicsEngine::setDieVelocity(int id, float lvx, float lvy, float lvz,
                    float avx, float avy, float avz) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.velocity = {lvx, lvy, lvz};
        b.angularVelocity = {avx, avy, avz};
        wake(b);
        break;
    }
}

void DicePhysicsEngine::setDieKinematic(int id, bool kinematic) {
    for (auto& b : bodies_) {
        if (b.id != id) continue;
        b.kinematic = kinematic;
        if (kinematic) {
            b.velocity = {};
            b.angularVelocity = {};
            b.sleeping = false;
            b.sleepTimer = 0.0f;
        } else {
            wake(b);
        }
        break;
    }
}

void DicePhysicsEngine::setContainerActive(bool active) { containerActive_ = active; }

void DicePhysicsEngine::setContainerPlanes(const std::vector<float>& flat) {
    containerPlanes_.clear();
    if (flat.size() % 4 != 0) return;
    const size_t count = std::min(flat.size() / 4, static_cast<size_t>(MAX_CONTAINER_PLANES));
    containerPlanes_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        Vec3 n{flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2]};
        const float len = n.length();
        if (len < 1e-6f) continue;
        n = n * (1.0f / len);
        containerPlanes_.push_back({n, flat[i * 4 + 3]});
    }
}

void DicePhysicsEngine::clearStatics() { statics_.clear(); staticCapacityDroppedCount_ = 0; }

bool DicePhysicsEngine::removeStatic(int userId) {
    const size_t before = statics_.size();
    statics_.erase(
        std::remove_if(statics_.begin(), statics_.end(),
            [userId](const StaticBody& s) { return s.userId == userId; }),
        statics_.end());
    return statics_.size() < before;
}

int DicePhysicsEngine::addStaticBox(int userId,
                 float cx, float cy, float cz,
                 float hx, float hy, float hz,
                 float qx, float qy, float qz, float qw,
                 int materialTag) {
    if (statics_.size() >= static_cast<size_t>(MAX_STATICS)) {
        ++staticCapacityDroppedCount_;
        return -1;
    }
    if (userId < 0) return -1;
    for (const auto& s : statics_) if (s.userId == userId) return -1;
    if (!positiveFinite(hx) || !positiveFinite(hy) || !positiveFinite(hz)) return -1;
    if (!allFinite(cx, cy, cz) || !allFinite(qx, qy, qz) || !std::isfinite(qw)) return -1;

    StaticBody body;
    body.userId = userId;
    body.shape = StaticShapeType::Box;
    body.center = {cx, cy, cz};
    body.rotation = Quat{qx, qy, qz, qw}.normalized();
    body.halfExtents = {hx, hy, hz};
    body.boundCenter = body.center;
    body.boundRadius = body.halfExtents.length();
    applyStaticMaterial(body, materialTag);
    body.hull.build({
        {-hx, -hy, -hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, {-hx,  hy, -hz},
        {-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz},
    });
    poseStaticHull(body);
    statics_.push_back(body);
    return userId;
}

int DicePhysicsEngine::addStaticPlane(int userId, float nx, float ny, float nz, float dist, int materialTag) {
    if (statics_.size() >= static_cast<size_t>(MAX_STATICS)) {
        ++staticCapacityDroppedCount_;
        return -1;
    }
    if (userId < 0) return -1;
    for (const auto& s : statics_) if (s.userId == userId) return -1;
    if (!allFinite(nx, ny, nz) || !std::isfinite(dist)) return -1;
    Vec3 n{nx, ny, nz};
    const float len = n.length();
    if (!(len >= 1e-6f) || !std::isfinite(len)) return -1;
    n = n * (1.0f / len);

    StaticBody body;
    body.userId = userId;
    body.shape = StaticShapeType::Plane;
    body.planeNormal = n;
    body.planeDist = dist;
    applyStaticMaterial(body, materialTag);
    statics_.push_back(body);
    return userId;
}

int DicePhysicsEngine::addStaticConvexHull(int userId,
                        float cx, float cy, float cz,
                        float qx, float qy, float qz, float qw,
                        const std::vector<float>& flatVerts,
                        int materialTag) {
    if (statics_.size() >= static_cast<size_t>(MAX_STATICS)) {
        ++staticCapacityDroppedCount_;
        return -1;
    }
    if (userId < 0) return -1;
    for (const auto& s : statics_) if (s.userId == userId) return -1;
    if (flatVerts.size() % 3 != 0) return -1;
    if (flatVerts.size() / 3 > MAX_VERTICES_PER_HULL) return -1;

    StaticBody body;
    body.userId = userId;
    body.shape = StaticShapeType::ConvexHull;
    body.center = {cx, cy, cz};
    body.rotation = Quat{qx, qy, qz, qw}.normalized();
    applyStaticMaterial(body, materialTag);

    std::vector<Vec3> verts;
    verts.reserve(flatVerts.size() / 3);
    for (size_t i = 0; i < flatVerts.size(); i += 3) {
        if (!allFinite(flatVerts[i], flatVerts[i + 1], flatVerts[i + 2])) return -1;
        verts.push_back({flatVerts[i], flatVerts[i + 1], flatVerts[i + 2]});
    }
    body.hull.build(verts);
    if (body.hull.verts.empty()) return -1;
    body.boundCenter = body.center + body.rotation.rotate((body.hull.aabbMin + body.hull.aabbMax) * 0.5f);
    body.boundRadius = ((body.hull.aabbMax - body.hull.aabbMin) * 0.5f).length();
    poseStaticHull(body);
    statics_.push_back(body);
    return userId;
}

int DicePhysicsEngine::addStaticOpenCylinder(int userId,
                          float cx, float cy, float cz,
                          float radius, float halfHeight,
                          int segments, bool closedBottom,
                          int materialTag) {
    if (statics_.size() >= static_cast<size_t>(MAX_STATICS)) {
        ++staticCapacityDroppedCount_;
        return -1;
    }
    if (userId < 0) return -1;
    for (const auto& s : statics_) if (s.userId == userId) return -1;
    if (!positiveFinite(radius) || !positiveFinite(halfHeight)) return -1;
    if (!allFinite(cx, cy, cz)) return -1;

    StaticBody body;
    body.userId = userId;
    body.shape = StaticShapeType::OpenCylinder;
    body.center = {cx, cy, cz};
    body.rotation = {0, 0, 0, 1};
    body.cylinderRadius = radius;
    body.cylinderHalfHeight = halfHeight;
    body.cylinderSegments = std::clamp(segments, 3, 32);
    body.cylinderClosedBottom = closedBottom;
    applyStaticMaterial(body, materialTag);
    statics_.push_back(body);
    return userId;
}

} // namespace dice_physics
