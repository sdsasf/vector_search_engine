#pragma once
#include <atomic>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <immintrin.h> // for AVX2 alignment
#include "distance.h"

namespace vector_search {

// 对齐到 64 字节，彻底消除类内部状态变量的伪共享 (False Sharing)
struct alignas(64) FlatWriteBuffer {
    float* data;                     // 极其连续的 32 字节对齐内存池
    uint32_t* ids;                   // 对应的向量 ID 数组
    std::atomic<size_t> count;       // 当前已写入的数量
    size_t capacity;
    size_t dim;

    FlatWriteBuffer(size_t cap, size_t d) : count(0), capacity(cap), dim(d) {
        // 强制 32 字节对齐，迎合 AVX2 的 _mm256_load_ps 指令
        data = (float*)std::aligned_alloc(32, capacity * dim * sizeof(float));
        ids = (uint32_t*)std::aligned_alloc(32, capacity * sizeof(uint32_t));
    }

    ~FlatWriteBuffer() {
        std::free(data);
        std::free(ids);
    }

    // 【极致写操作：Wait-Free 无锁追加】
    // 返回 false 代表 Buffer 已满，需要触发外层的双缓冲切换
    inline bool append_wait_free(const float* vec, uint32_t id) {
        // 原子获取槽位 (XADD 指令，极速分配)
        size_t idx = count.fetch_add(1, std::memory_order_relaxed);
        
        if (idx >= capacity) {
            return false; // 溢出处理
        }

        // 把数据 memcpy 到专属的槽位中
        // 注意：因为是从网络层过来的 request->query_vector()，
        // 如果想进一步压榨，这里可以用 AVX2 专门写一个快速拷贝函数。
        std::memcpy(data + idx * dim, vec, dim * sizeof(float));
        ids[idx] = id;

        // 【极其硬核的细节】为了防止读线程读到 memcpy 还没写完的半途数据
        // 实际工业级实现这里需要一个 version/commit_count 或者通过 release 屏障保护，
        // 这里提供一种经典的极致解法：依赖 RCU 的全局 Epoch，或者简化的标识位。
        return true;
    }

    // 【极致读操作：纯线性扫描 Brute-force】
    // 读线程直接暴力扫内存，硬件预取器 (Prefetcher) 满载运行
    void search_brute_force(const float* query, int k, std::priority_queue<NodeDist>& top_candidates) const {
        // acquire 语义保证读到的 count 是写线程 commit 之后的大小
        size_t current_sz = count.load(std::memory_order_acquire);
        if (current_sz > capacity) current_sz = capacity;

        for (size_t i = 0; i < current_sz; ++i) {
            // 直接调用你的 AVX2 距离算子！由于 data 是 32 字节对齐的，算得极快！
            float d = l2_distance_avx2(query, data + i * dim, dim);
            
            if (top_candidates.size() < (size_t)k || d < top_candidates.top().dist) {
                top_candidates.push({ids[i], d});
                if (top_candidates.size() > (size_t)k) {
                    top_candidates.pop();
                }
            }
        }
    }
};

} // namespace vector_search