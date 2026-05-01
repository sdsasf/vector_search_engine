#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <immintrin.h>

#include "distance.h"

namespace vector_search {

struct alignas(64) FlatWriteBuffer {
    float* data;
    uint32_t* ids;
    std::atomic<uint8_t>* ready;
    std::atomic<size_t> count;
    size_t capacity;
    size_t dim;

    FlatWriteBuffer(size_t cap, size_t d) : data(nullptr), ids(nullptr), ready(nullptr), count(0), capacity(cap), dim(d) {
        const size_t data_bytes = capacity * dim * sizeof(float);
        const size_t aligned_data_bytes = ((data_bytes + 31) / 32) * 32;
        const size_t ids_bytes = capacity * sizeof(uint32_t);
        const size_t aligned_ids_bytes = ((ids_bytes + 31) / 32) * 32;
        data = static_cast<float*>(std::aligned_alloc(32, aligned_data_bytes));
        ids = static_cast<uint32_t*>(std::aligned_alloc(32, aligned_ids_bytes));
        ready = new std::atomic<uint8_t>[capacity];
        for (size_t i = 0; i < capacity; ++i) {
            ready[i].store(0, std::memory_order_relaxed);
        }
    }

    ~FlatWriteBuffer() {
        delete[] ready;
        std::free(data);
        std::free(ids);
    }

    inline bool append_wait_free(const float* vec, uint32_t id) {
        size_t idx = count.fetch_add(1, std::memory_order_relaxed);
        if (idx >= capacity) {
            return false;
        }

        std::memcpy(data + idx * dim, vec, dim * sizeof(float));
        ids[idx] = id;
        ready[idx].store(1, std::memory_order_release);
        return true;
    }

    inline bool is_ready(size_t idx) const {
        return idx < capacity && ready[idx].load(std::memory_order_acquire) != 0;
    }

    inline size_t visible_count() const {
        size_t current = count.load(std::memory_order_acquire);
        return current > capacity ? capacity : current;
    }

    void search_brute_force(const float* query, int k, std::priority_queue<NodeDist>& top_candidates) const {
        size_t current_sz = visible_count();
        for (size_t i = 0; i < current_sz; ++i) {
            if (!is_ready(i)) continue;

            float d = l2_distance_avx2(query, data + i * dim, dim);
            if (top_candidates.size() < static_cast<size_t>(k) || d < top_candidates.top().dist) {
                top_candidates.push({ids[i], d});
                if (top_candidates.size() > static_cast<size_t>(k)) {
                    top_candidates.pop();
                }
            }
        }
    }
};

} // namespace vector_search
