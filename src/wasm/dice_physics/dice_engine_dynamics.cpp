/**
 * dice_engine_dynamics.cpp — Dynamic (non-die) rigid-body props: lifecycle,
 * integration, collision against dice/statics/table/container, and the
 * dynamic×die / dynamic×dynamic contact solver.
 *
 * Dynamic props live in their own `dynamics_` vector, separate from dice's
 * `bodies_`. This mirrors the static-collider lifecycle (caller-supplied id,
 * capacity-checked vector) while participating in the solver like a die
 * (integrated, collides, sleeps).
 *
 * Broadphase: since Phase 8 (docs/WASM_ENGINE.md), die-dynamic and
 * dynamic-dynamic pairs are grid-broadphased rather than brute-forced.
 * Dynamics get their own CSR bucket grid (`dynGrid_`, rebuilt each substep by
 * `rebuildBroadphase` in dice_engine_collision_dynamic.cpp) sharing the die
 * grid's dimensions and origin; the templated `forEachDieDynamicPair` /
 * `forEachDynamicPair` in dice_physics_engine.hpp walk it.
 *
 * Both walkers still keep a nested-loop path, taken when `useBroadphase_` is
 * false (or a population is too small to pair). `useBroadphase_` is a test
 * hook -- `setBroadphaseForTesting`, used to diff grid pairs against
 * brute-force ones -- so production always takes the grid path, but the
 * fallback is real code and not dead.
 *
 * Either way `MAX_DYNAMICS` (256) is a memory / event-budget cap, not a
 * brute-force-cost cap: it was raised 64 -> 256 *because* the grid made the
 * pair cost sub-quadratic. Do not reintroduce nested loops on the production
 * path on the assumption that the cap keeps them cheap.
 */

#include "../dice_physics_engine.hpp"

#include <algorithm>
#include <cmath>

namespace dice_physics {

// ---------------------------------------------------------------------------
// Shared helpers (mirrors of the RigidBody-typed ones in
// dice_engine_collision_static.cpp / dice_engine_integrate.cpp, duplicated
// rather than templated to keep the well-tested die code paths untouched).
// ---------------------------------------------------------------------------

float DicePhysicsEngine::inertiaScalar(const DynamicBody& b) {
    if (b.hull.verts.empty()) {
        return 0.4f * b.mass * b.radius * b.radius;
    }
    Vec3 dim = b.hull.aabbMax - b.hull.aabbMin;
    float ix = (1.0f / 12.0f) * b.mass * (dim.y * dim.y + dim.z * dim.z);
    float iy = (1.0f / 12.0f) * b.mass * (dim.x * dim.x + dim.z * dim.z);
    float iz = (1.0f / 12.0f) * b.mass * (dim.x * dim.x + dim.y * dim.y);
    return (ix + iy + iz) / 3.0f;
}

CollisionEvent DicePhysicsEngine::makeEvent(
    const DynamicBody& primary,
    int otherId,
    float impactSpeed,
    float linearSpeedSq,
    float angularSpeedSq,
    int staticColliderId,
    int materialTag
) {
    return {
        dynamicEventOtherId(primary.userId),
        otherId,
        impactSpeed,
        primary.mass,
        inertiaScalar(primary),
        linearSpeedSq >= 0.0f ? linearSpeedSq : primary.velocity.lengthSq(),
        angularSpeedSq >= 0.0f ? angularSpeedSq : primary.angularVelocity.lengthSq(),
        staticColliderId,
        materialTag,
    };
}

int DicePhysicsEngine::dynamicEventOtherId(int userId) {
    return DYNAMIC_EVENT_ID_BASE - userId;
}

void DicePhysicsEngine::wake(DynamicBody& b) {
    b.sleeping = false;
    b.sleepTimer = 0.0f;
}

void DicePhysicsEngine::integrateDynamic(DynamicBody& b, float dt) {
    if (b.kinematic) return;
    b.velocity.y += gravity_ * dt;
    b.velocity = b.velocity * std::exp(-LINEAR_DAMPING * dt);
    b.angularVelocity = b.angularVelocity * std::exp(-ANGULAR_DAMPING * dt);
    const float lin = b.velocity.length();
    if (lin > MAX_LINEAR_SPEED) b.velocity = b.velocity * (MAX_LINEAR_SPEED / lin);
    const float ang = b.angularVelocity.length();
    if (ang > MAX_ANGULAR_SPEED) b.angularVelocity = b.angularVelocity * (MAX_ANGULAR_SPEED / ang);
    b.position += b.velocity * dt;
    b.rotation = b.rotation.integrate(b.angularVelocity, dt);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void DicePhysicsEngine::clearDynamics() {
    dynamics_.clear();
    dynamicCapacityDroppedCount_ = 0;
}

bool DicePhysicsEngine::removeDynamic(int userId) {
    const size_t before = dynamics_.size();
    dynamics_.erase(
        std::remove_if(dynamics_.begin(), dynamics_.end(),
            [userId](const DynamicBody& d) { return d.userId == userId; }),
        dynamics_.end());
    return dynamics_.size() < before;
}

void DicePhysicsEngine::setDynamicKinematic(int userId, bool kinematic) {
    for (auto& d : dynamics_) {
        if (d.userId != userId) continue;
        d.kinematic = kinematic;
        if (kinematic) {
            d.velocity = {};
            d.angularVelocity = {};
            d.sleeping = false;
            d.sleepTimer = 0.0f;
        } else {
            wake(d);
        }
        break;
    }
}

void DicePhysicsEngine::setDynamicTransform(int userId, float px, float py, float pz,
                                            float qx, float qy, float qz, float qw) {
    for (auto& d : dynamics_) {
        if (d.userId != userId) continue;
        d.position = {px, py, pz};
        d.rotation = Quat{qx, qy, qz, qw}.normalized();
        d.velocity = {};
        d.angularVelocity = {};
        wake(d);
        break;
    }
}

void DicePhysicsEngine::setDynamicVelocity(int userId, float lvx, float lvy, float lvz,
                                           float avx, float avy, float avz) {
    for (auto& d : dynamics_) {
        if (d.userId != userId) continue;
        d.velocity = {lvx, lvy, lvz};
        d.angularVelocity = {avx, avy, avz};
        wake(d);
        break;
    }
}

void DicePhysicsEngine::applyDynamicImpulse(int userId, float fx, float fy, float fz) {
    for (auto& d : dynamics_) {
        if (d.userId != userId) continue;
        d.velocity += Vec3{fx, fy, fz} * d.invMass;
        wake(d);
        break;
    }
}

void DicePhysicsEngine::applyDynamicTorqueImpulse(int userId, float tx, float ty, float tz) {
    for (auto& d : dynamics_) {
        if (d.userId != userId) continue;
        d.angularVelocity += d.applyInvInertiaWorld(Vec3{tx, ty, tz});
        wake(d);
        break;
    }
}

int DicePhysicsEngine::addDynamicBox(int userId, float mass,
                                     float cx, float cy, float cz,
                                     float hx, float hy, float hz,
                                     float qx, float qy, float qz, float qw,
                                     int materialTag) {
    if (dynamics_.size() >= static_cast<size_t>(MAX_DYNAMICS)) {
        ++dynamicCapacityDroppedCount_;
        return -1;
    }
    if (userId < 0) return -1;
    for (const auto& d : dynamics_) if (d.userId == userId) return -1;
    if (!positiveFinite(mass) || !positiveFinite(hx) || !positiveFinite(hy) || !positiveFinite(hz)) return -1;
    if (!allFinite(cx, cy, cz) || !allFinite(qx, qy, qz) || !std::isfinite(qw)) return -1;

    DynamicBody body;
    body.userId = userId;
    body.shape = DynamicShapeType::Box;
    body.materialTag = static_cast<uint8_t>(std::clamp(materialTag, 0, 255));
    body.position = {cx, cy, cz};
    body.rotation = Quat{qx, qy, qz, qw}.normalized();
    body.halfExtents = {hx, hy, hz};
    body.mass = mass;
    body.invMass = 1.0f / mass;
    body.radius = Vec3{hx, hy, hz}.length();
    body.hull.build({
        {-hx, -hy, -hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, {-hx,  hy, -hz},
        {-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz},
    });
    body.computeInertiaFromHull();
    dynamics_.push_back(body);
    reserveContactStorage();
    return userId;
}

int DicePhysicsEngine::addDynamicHull(int userId, float mass,
                                      float cx, float cy, float cz,
                                      float qx, float qy, float qz, float qw,
                                      const std::vector<float>& flatVerts,
                                      int materialTag) {
    if (dynamics_.size() >= static_cast<size_t>(MAX_DYNAMICS)) {
        ++dynamicCapacityDroppedCount_;
        return -1;
    }
    if (userId < 0) return -1;
    for (const auto& d : dynamics_) if (d.userId == userId) return -1;
    if (!positiveFinite(mass)) return -1;
    if (!allFinite(cx, cy, cz) || !allFinite(qx, qy, qz) || !std::isfinite(qw)) return -1;
    if (flatVerts.size() % 3 != 0) return -1;
    if (flatVerts.size() / 3 > MAX_VERTICES_PER_HULL) return -1;

    std::vector<Vec3> verts;
    verts.reserve(flatVerts.size() / 3);
    for (size_t i = 0; i < flatVerts.size(); i += 3) {
        if (!allFinite(flatVerts[i], flatVerts[i + 1], flatVerts[i + 2])) return -1;
        verts.push_back({flatVerts[i], flatVerts[i + 1], flatVerts[i + 2]});
    }
    if (verts.empty()) return -1;

    DynamicBody body;
    body.userId = userId;
    body.shape = DynamicShapeType::Hull;
    body.materialTag = static_cast<uint8_t>(std::clamp(materialTag, 0, 255));
    body.position = {cx, cy, cz};
    body.rotation = Quat{qx, qy, qz, qw}.normalized();
    body.mass = mass;
    body.invMass = 1.0f / mass;
    body.hull.build(verts);
    if (body.hull.verts.empty()) return -1;
    body.radius = std::max((body.hull.aabbMax - body.hull.aabbMin).length() * 0.5f, 0.05f);
    body.computeInertiaFromHull();
    dynamics_.push_back(body);
    reserveContactStorage();
    return userId;
}

int DicePhysicsEngine::getDynamicCount() const { return static_cast<int>(dynamics_.size()); }

uint32_t DicePhysicsEngine::getDynamicCapacityDroppedCount() const { return dynamicCapacityDroppedCount_; }

const std::vector<float>& DicePhysicsEngine::buildDynamicTransformBuffer() {
    dynamicTransformBuffer_.clear();
    dynamicTransformBuffer_.reserve(dynamics_.size() * 7);
    for (const auto& d : dynamics_) {
        dynamicTransformBuffer_.push_back(d.position.x);
        dynamicTransformBuffer_.push_back(d.position.y);
        dynamicTransformBuffer_.push_back(d.position.z);
        dynamicTransformBuffer_.push_back(d.rotation.x);
        dynamicTransformBuffer_.push_back(d.rotation.y);
        dynamicTransformBuffer_.push_back(d.rotation.z);
        dynamicTransformBuffer_.push_back(d.rotation.w);
    }
    return dynamicTransformBuffer_;
}

const std::vector<float>& DicePhysicsEngine::buildDynamicIdBuffer() {
    dynamicIdBuffer_.clear();
    dynamicIdBuffer_.reserve(dynamics_.size());
    for (const auto& d : dynamics_) {
        dynamicIdBuffer_.push_back(static_cast<float>(d.userId));
    }
    return dynamicIdBuffer_;
}

} // namespace dice_physics
