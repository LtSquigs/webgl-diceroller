/**
 * dice_engine_collision_dynamic.cpp — Uniform-grid broadphase: grid
 * dimensions, cell indexing, and the per-substep CSR bucket build. Pair
 * enumeration lives in dice_physics_engine.hpp (templated forEachDiePair and
 * friends); contact generation and the sequential-impulse solve live in
 * dice_engine_solver.cpp.
 */

#include "../dice_physics_engine.hpp"

#include <algorithm>
#include <cmath>

namespace dice_physics {

namespace {

// floor(rel / cell) clamped to [0, maxIndex], clamped *before* the int cast:
// a NaN/Inf or far-off position (a forged snapshot, a degenerate collider)
// must not hit the float->int UB the cast would otherwise be. Identical to
// clamp(int(floor(...))) for every in-range value, so pair order is unchanged.
int cellIndex(float rel, float cell, int maxIndex) {
    const float f = std::floor(rel / cell);
    if (!(f >= 0.0f)) return 0;
    if (f >= static_cast<float>(maxIndex)) return maxIndex;
    return static_cast<int>(f);
}

} // namespace

void DicePhysicsEngine::ensureDieGridDimensions() {
    if (gridCols_ > 0) return;
    gridOriginX_ = -tableHalfW_ - GRID_CELL_SIZE;
    gridOriginZ_ = -tableHalfD_ - GRID_CELL_SIZE;
    const float extentX = 2.0f * tableHalfW_ + 2.0f * GRID_CELL_SIZE;
    const float extentZ = 2.0f * tableHalfD_ + 2.0f * GRID_CELL_SIZE;
    gridCols_ = std::max(1, static_cast<int>(std::ceil(extentX / GRID_CELL_SIZE)));
    gridRows_ = std::max(1, static_cast<int>(std::ceil(extentZ / GRID_CELL_SIZE)));
}

int DicePhysicsEngine::bodyCellXMin(float x, float radius) const {
    const float rel = x - radius - gridOriginX_;
    return cellIndex(rel, GRID_CELL_SIZE, gridCols_ - 1);
}

int DicePhysicsEngine::bodyCellXMax(float x, float radius) const {
    const float rel = x + radius - gridOriginX_;
    return cellIndex(rel, GRID_CELL_SIZE, gridCols_ - 1);
}

int DicePhysicsEngine::bodyCellZMin(float z, float radius) const {
    const float rel = z - radius - gridOriginZ_;
    return cellIndex(rel, GRID_CELL_SIZE, gridRows_ - 1);
}

int DicePhysicsEngine::bodyCellZMax(float z, float radius) const {
    const float rel = z + radius - gridOriginZ_;
    return cellIndex(rel, GRID_CELL_SIZE, gridRows_ - 1);
}

template <typename BodyVec>
void DicePhysicsEngine::rebuildGrid(BroadphaseGrid& grid, const BodyVec& bodies, float expand) {
    const size_t cellCount = static_cast<size_t>(gridCols_) * static_cast<size_t>(gridRows_);
    const size_t n = bodies.size();
    grid.start.assign(cellCount + 1, 0);
    grid.range.resize(n * 4);

    // Reserve for the widest footprint any body can have at the largest
    // expand the contact pass ever uses, so items never grows mid-step.
    const auto cols = static_cast<size_t>(gridCols_);
    const auto rows = static_cast<size_t>(gridRows_);
    size_t bound = 0;
    for (const auto& b : bodies) {
        const float reach = 2.0f * (b.radius + std::max(expand, SPECULATIVE_MAX));
        const float spanF = std::floor(reach / GRID_CELL_SIZE) + 2.0f;
        // `!(spanF < limit)` also catches NaN (a body whose radius or the
        // expand went non-finite): casting NaN to size_t is UB, so such a
        // body is bounded by the whole grid like any oversized one.
        const size_t span = !(spanF < static_cast<float>(std::max(cols, rows)))
                                ? std::max(cols, rows)
                                : static_cast<size_t>(spanF);
        bound += std::min(span, cols) * std::min(span, rows);
    }
    if (grid.items.capacity() < bound) grid.items.reserve(bound);

    for (size_t i = 0; i < n; ++i) {
        const auto& b = bodies[i];
        const float r = b.radius + expand;
        int32_t* range = grid.range.data() + i * 4;
        range[0] = bodyCellXMin(b.position.x, r);
        range[1] = bodyCellXMax(b.position.x, r);
        range[2] = bodyCellZMin(b.position.z, r);
        range[3] = bodyCellZMax(b.position.z, r);
        for (int cz = range[2]; cz <= range[3]; ++cz) {
            for (int cx = range[0]; cx <= range[1]; ++cx) {
                ++grid.start[static_cast<size_t>(cz) * cols + static_cast<size_t>(cx) + 1];
            }
        }
    }
    for (size_t c = 0; c < cellCount; ++c) grid.start[c + 1] += grid.start[c];
    grid.items.resize(grid.start[cellCount]);
    grid.cursor.assign(grid.start.begin(), grid.start.end() - 1);
    // Bodies in ascending index order, so each cell lists them ascending.
    for (size_t i = 0; i < n; ++i) {
        const int32_t* range = grid.range.data() + i * 4;
        for (int cz = range[2]; cz <= range[3]; ++cz) {
            for (int cx = range[0]; cx <= range[1]; ++cx) {
                const size_t c = static_cast<size_t>(cz) * cols + static_cast<size_t>(cx);
                grid.items[grid.cursor[c]++] = static_cast<uint32_t>(i);
            }
        }
    }
}

void DicePhysicsEngine::rebuildBroadphase(float expand) {
    ensureDieGridDimensions();
    gridExpand_ = expand;
    if (!useBroadphase_) return;
    rebuildGrid(dieGrid_, bodies_, expand);
    rebuildGrid(dynGrid_, dynamics_, expand);
    if (partnerScratch_.capacity() < std::max(bodies_.size(), dynamics_.size())) {
        partnerScratch_.reserve(std::max(bodies_.size(), dynamics_.size()));
    }
}

} // namespace dice_physics
