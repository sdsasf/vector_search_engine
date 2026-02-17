#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "ebr_manager.h" // 【修复】引入 EBR 管理器

namespace vector_search {

// 硬件缓存行大小对齐，消除伪共享
constexpr std::size_t CACHE_LINE_SIZE = 64; // 通常为 64 字节
constexpr int MAX_HNSW_LEVELS = 16; // 【修复】定义最大层数

struct NeighborList {
    uint32_t count;
    uint32_t capacity;
    uint32_t neighbors[]; // 柔性数组
};

// 高效自旋锁，适用于短时间的锁持有，专用于原地高频更新
struct SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;

    inline void lock() {
        // test_and_set() 映射到底层的 XCHG 指令
        while (locked.test_and_set(std::memory_order_acquire)) {
            // _mm_pause() 告诉 CPU 这是一个自旋循环，防止流水线乱序执行引发的惩罚
            _mm_pause(); 
        }
    }

    inline void unlock() {
        locked.clear(std::memory_order_release);
    }
};

struct alignas(CACHE_LINE_SIZE) HnswNode {
    const float* vector_data; 
    
    // 【修复】改为多层结构：每个楼层都有自己独立的并发邻居表指针
    std::atomic<NeighborList*> neighbor_lists[MAX_HNSW_LEVELS];

    int level; // 该节点所在的最高层数
    SpinLock node_lock; // 保护节点状态的自旋锁

    // 初始化节点
    void init(const float* data, int max_level) {
        vector_data = data;
        level = max_level;
        for (int i = 0; i < MAX_HNSW_LEVELS; ++i) {
            neighbor_lists[i].store(nullptr, std::memory_order_relaxed);
        }
    }


    // 【修复】接收 layer 参数，获取指定层的邻居表
    NeighborList* get_neighbors_rcu(int layer) const {
        if (layer >= MAX_HNSW_LEVELS) return nullptr;
        return neighbor_lists[layer].load(std::memory_order_acquire);
    }

    // 【修复】接收 layer 参数，往指定层添加邻居
    void add_neighbor_rcu(int layer, uint32_t new_neighbor_id) {
        if (layer >= MAX_HNSW_LEVELS) return;

        NeighborList* old_list = neighbor_lists[layer].load(std::memory_order_relaxed); 
        
        while (true) {
            size_t new_capacity = (old_list == nullptr) ? 1 : old_list->count + 1;
            size_t alloc_size = sizeof(NeighborList) + new_capacity * sizeof(uint32_t);
            NeighborList* new_list = (NeighborList*)std::malloc(alloc_size); 
            
            new_list->capacity = new_capacity;
            new_list->count = (old_list == nullptr) ? 0 : old_list->count;
            
            if (old_list != nullptr) {
                std::memcpy(new_list->neighbors, old_list->neighbors, old_list->count * sizeof(uint32_t));
            }
            
            new_list->neighbors[new_list->count++] = new_neighbor_id;

            if (neighbor_lists[layer].compare_exchange_weak(
                    old_list, new_list, 
                    std::memory_order_release, 
                    std::memory_order_relaxed)) {
                
                if (old_list != nullptr) {
                    EBRManager::get_instance().defer_free(old_list);
                }
                break;
            } else {
                std::free(new_list);
            }
        }
    }
};

} // namespace vector_search