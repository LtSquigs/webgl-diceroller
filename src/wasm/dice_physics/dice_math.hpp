/**
 * dice_math.hpp — Vector/quaternion/matrix primitives and convex hull helper.
 *
 * Part of the dice_physics_engine module split; included by
 * dice_physics_engine.hpp.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace dice_physics {

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

/// True for a finite value strictly greater than zero. Collider registration
/// uses this instead of `v <= 0.0f` because NaN passes that test: a NaN
/// radius or half-extent then reaches the broadphase, where
/// `static_cast<size_t>(NaN)` is undefined behaviour (libFuzzer found it).
inline bool positiveFinite(float v) { return std::isfinite(v) && v > 0.0f; }

/// True when every component is finite.
inline bool allFinite(float a, float b, float c) {
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(c);
}

struct Vec3 {
    float x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s)        const { return {x * s,   y * s,   z * s};   }
    Vec3 operator/(float s)        const { return {x / s,   y / s,   z / s};   }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

    static float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
    }
    float lengthSq() const { return x*x + y*y + z*z; }
    float length()   const { return std::sqrt(lengthSq()); }
    Vec3 normalized() const {
        float l = length();
        return l < 1e-7f ? Vec3{} : *this * (1.0f / l);
    }
};

struct Quat {
    float x = 0, y = 0, z = 0, w = 1;

    Quat() = default;
    Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    Quat operator*(const Quat& o) const {
        return {
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w,
            w*o.w - x*o.x - y*o.y - z*o.z
        };
    }
    Quat conjugate() const { return {-x, -y, -z, w}; }
    Quat normalized() const {
        float l = std::sqrt(x*x + y*y + z*z + w*w);
        return l < 1e-7f ? Quat{0,0,0,1} : Quat{x/l, y/l, z/l, w/l};
    }
    Vec3 rotate(const Vec3& v) const {
        Vec3 qv = {x, y, z};
        Vec3 uv  = Vec3::cross(qv, v);
        Vec3 uuv = Vec3::cross(qv, uv);
        return v + (uv * (2.0f * w)) + (uuv * 2.0f);
    }
    Quat integrate(const Vec3& omega, float dt) const {
        float hx = omega.x * 0.5f * dt;
        float hy = omega.y * 0.5f * dt;
        float hz = omega.z * 0.5f * dt;
        Quat dq = {
            hx*w + hy*z - hz*y,
            hy*w + hz*x - hx*z,
            hz*w + hx*y - hy*x,
           -hx*x - hy*y - hz*z
        };
        return Quat{x+dq.x, y+dq.y, z+dq.z, w+dq.w}.normalized();
    }
};

struct Mat3 {
    float m[9];
    static Mat3 identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
    static Mat3 diagonal(float ix, float iy, float iz) {
        return { ix,0,0, 0,iy,0, 0,0,iz };
    }
    Vec3 mul(const Vec3& v) const {
        return {
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        };
    }
};

/** Orthonormal tangent pair for a unit normal (deterministic, branch-stable). */
inline void tangentBasis(const Vec3& n, Vec3& t1, Vec3& t2) {
    if (std::abs(n.x) >= 0.57735027f) {
        t1 = Vec3{n.y, -n.x, 0.0f}.normalized();
    } else {
        t1 = Vec3{0.0f, n.z, -n.y}.normalized();
    }
    t2 = Vec3::cross(n, t1);
}

inline uint64_t fnv1a64(const uint8_t* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ---------------------------------------------------------------------------
// PolyHull
// ---------------------------------------------------------------------------

struct PolyHull {
    std::vector<Vec3> verts;
    std::vector<Vec3> faceNormals;
    std::vector<Vec3> edgeDirs;
    Vec3 aabbMin, aabbMax;

    void build(const std::vector<Vec3>& input) {
        verts = input;
        if (verts.empty()) return;

        aabbMin = aabbMax = verts[0];
        for (size_t i = 1; i < verts.size(); ++i) {
            aabbMin.x = std::min(aabbMin.x, verts[i].x);
            aabbMin.y = std::min(aabbMin.y, verts[i].y);
            aabbMin.z = std::min(aabbMin.z, verts[i].z);
            aabbMax.x = std::max(aabbMax.x, verts[i].x);
            aabbMax.y = std::max(aabbMax.y, verts[i].y);
            aabbMax.z = std::max(aabbMax.z, verts[i].z);
        }

        computeFaceNormals();
        computeEdgeDirections();
    }

    void computeFaceNormals() {
        const int n = static_cast<int>(verts.size());
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                for (int k = j+1; k < n; ++k) {
                    Vec3 e1 = verts[j] - verts[i];
                    Vec3 e2 = verts[k] - verts[i];
                    Vec3 nrm = Vec3::cross(e1, e2).normalized();
                    if (nrm.lengthSq() < 1e-6f) continue;

                    float d = Vec3::dot(nrm, verts[i]);
                    int pos = 0, neg = 0;
                    for (int m = 0; m < n; ++m) {
                        if (m == i || m == j || m == k) continue;
                        float side = Vec3::dot(nrm, verts[m]) - d;
                        if (side > 1e-4f) pos++;
                        if (side < -1e-4f) neg++;
                    }
                    if (pos > 0 && neg > 0) continue;

                    if (neg > 0) nrm = nrm * -1.0f;
                    bool exists = false;
                    for (const auto& fn : faceNormals) {
                        if (Vec3::dot(fn, nrm) > 0.995f) { exists = true; break; }
                    }
                    if (!exists) faceNormals.push_back(nrm);
                }
            }
        }
    }

    void computeEdgeDirections() {
        const int n = static_cast<int>(verts.size());
        float minDistSq = 1e20f;
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                float d = (verts[j] - verts[i]).lengthSq();
                if (d > 1e-6f && d < minDistSq) minDistSq = d;
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                float d = (verts[j] - verts[i]).lengthSq();
                if (std::abs(d - minDistSq) < 1e-3f) {
                    Vec3 dir = (verts[j] - verts[i]).normalized();
                    bool exists = false;
                    for (const auto& e : edgeDirs) {
                        if (std::abs(Vec3::dot(e, dir)) > 0.995f) { exists = true; break; }
                    }
                    if (!exists) edgeDirs.push_back(dir);
                }
            }
        }
    }

    /**
     * Radius of the largest sphere about the hull's own origin that the hull
     * strictly contains: min over face planes of the support distance along
     * that face normal.
     *
     * The swept-contact path (see CCD_MOTION_FRACTION) uses it as a sweep
     * proxy — a sphere this size can be swept in place of the hull without
     * ever reporting a hit the hull itself would not have had, and unlike the
     * bounding radius it does not depend on the hull's orientation, which is
     * exactly what a proxy for a tumbling die needs. 0 when the hull has no
     * usable faces.
     */
    float inscribedRadius() const {
        if (verts.empty() || faceNormals.empty()) return 0.0f;
        float inscribed = 1e30f;
        for (const auto& n : faceNormals) {
            float support = -1e30f;
            for (const auto& v : verts) support = std::max(support, Vec3::dot(n, v));
            inscribed = std::min(inscribed, support);
        }
        return (inscribed > 0.0f && std::isfinite(inscribed)) ? inscribed : 0.0f;
    }
};

} // namespace dice_physics
