#pragma once
#include <queue>
#include <vector>
#include <limits>
#include <cmath>
#include <random>
#include <mutex>
#include <algorithm>
#include <immintrin.h>
#include "distance.h"
#include "hnsw_node.h"

namespace vector_search {

struct NodeDist {
    uint32_t id;
    float dist;
    bool operator<(const NodeDist& other) const { return dist < other.dist; }
    bool operator>(const NodeDist& other) const { return dist > other.dist; }
};

class HnswIndex {
public:
    // 初始化参数：维度、最大承载量、每层最大邻居数 M、建图搜索深度 ef_construction
    HnswIndex(size_t dim, size_t max_elements, int M = 16, int ef_construction = 100)
        : dim_(dim), max_elements_(max_elements), M_(M), ef_construction_(ef_construction) {
        
        // 1. 核心存储：一次性向操作系统申请巨大的、按 64 字节对齐的连续内存块。
        // 这样彻底杜绝了动态扩容带来的指针失效问题，并且最大化 L1/L2 Cache 命中率。
        nodes_ = (HnswNode*)std::aligned_alloc(CACHE_LINE_SIZE, max_elements_ * sizeof(HnswNode));
        
        // HNSW 层数概率分布因子
        level_mult_ = 1.0 / std::log(1.0 * M_);
        
        // 全局状态初始化
        enter_point_id_.store(0, std::memory_order_relaxed);
        max_level_.store(-1, std::memory_order_relaxed);
    }

    ~HnswIndex() {
        std::free(nodes_);
    }

    // O(1) 极速获取节点指针
    inline HnswNode* get_node(uint32_t id) {
        return &nodes_[id];
    }

    // ==========================================
    // 核心建图方法：支持多线程高并发调用
    // ==========================================
    void insert(const float* vector_data, uint32_t id) {
        auto& ebr = EBRManager::get_instance();
        ebr.enter_rcu_read(); // 建图过程涉及大量读图操作，必须受 RCU 保护

        // 1. 初始化新节点
        int new_node_level = get_random_level();
        HnswNode* new_node = get_node(id);
        new_node->init(vector_data, new_node_level);

        int curr_max_level = max_level_.load(std::memory_order_acquire);

        // 2. 处理冷启动（插入的是整张图的第一个节点）
        if (curr_max_level == -1) {
            std::lock_guard<std::mutex> lock(ep_mutex_); // 冷启动加锁是安全的
            if (max_level_.load(std::memory_order_acquire) == -1) {
                enter_point_id_.store(id, std::memory_order_release);
                max_level_.store(new_node_level, std::memory_order_release);
                ebr.exit_rcu_read();
                return;
            }
            curr_max_level = max_level_.load(std::memory_order_acquire);
        }

        uint32_t curr_obj = enter_point_id_.load(std::memory_order_acquire);
        float curr_dist = l2_distance_avx2(vector_data, get_node(curr_obj)->vector_data, dim_);

        // 3. 阶段一：找到新节点该插入的目标层的入口（垂直降落）
        for (int level = curr_max_level; level > new_node_level; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                NeighborList* neighbors = get_node(curr_obj)->get_neighbors_rcu(level);
                if (!neighbors) continue;

                for (uint32_t i = 0; i < neighbors->count; ++i) {
                    uint32_t candidate_id = neighbors->neighbors[i];
                    float d = l2_distance_avx2(vector_data, get_node(candidate_id)->vector_data, dim_);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = candidate_id;
                        changed = true;
                    }
                }
            }
        }

        // 4. 阶段二：逐层寻找最近邻并建立双向连接
        int min_level = std::min(curr_max_level, new_node_level);
        for (int level = min_level; level >= 0; --level) {
            // 在当前层寻找离新节点最近的 ef_construction 个邻居
            auto top_candidates = search_layer(vector_data, curr_obj, ef_construction_, level);
            
            // 挑选最近的 M 个建立双向边 (RCU 保证了无锁并发写)
            int num_to_connect = std::min((int)top_candidates.size(), M_);
            for (int i = 0; i < num_to_connect; ++i) {
                uint32_t neighbor_id = top_candidates[i];
                
                // 无锁添加边：新节点 -> 邻居
                new_node->add_neighbor_rcu(level, neighbor_id);
                // 无锁添加边：邻居 -> 新节点
                get_node(neighbor_id)->add_neighbor_rcu(level, id);
            }
            
            // 准备进入下一层，用本层找到的最近点作为下层搜索的起点
            if (!top_candidates.empty()) {
                curr_obj = top_candidates[0]; 
            }
        }

        // 5. 阶段三：如果新节点运气爆棚，层数突破了天际，更新全局最高入口
        if (new_node_level > curr_max_level) {
            std::lock_guard<std::mutex> lock(ep_mutex_); // 极低频操作，锁竞争可忽略
            if (new_node_level > max_level_.load(std::memory_order_acquire)) {
                enter_point_id_.store(id, std::memory_order_release);
                max_level_.store(new_node_level, std::memory_order_release);
            }
        }

        ebr.exit_rcu_read();
    }

    // 专供 Bulk Load (冷启动全量导入) 使用的上帝模式接口
    // 特性：0 次 EBR 操作，0 次内存 Copy，纯自旋锁原地并发更新
    void insert_bulk(const float* vector_data, uint32_t id) {
        // 1. 初始化新节点
        int new_node_level = get_random_level();
        HnswNode* new_node = get_node(id);
        
        // 注意：底层 init 函数里，最好一次性把各层的 NeighborList 数组 
        // 预分配到 M_ (第0层为 M0_) 的最大容量，避免后续触发扩容
        new_node->init(vector_data, new_node_level);

        int curr_max_level = max_level_.load(std::memory_order_acquire);

        // 2. 处理冷启动（插入的是整张图的第一个节点）
        if (curr_max_level == -1) {
            std::lock_guard<std::mutex> lock(ep_mutex_); 
            if (max_level_.load(std::memory_order_acquire) == -1) {
                enter_point_id_.store(id, std::memory_order_release);
                max_level_.store(new_node_level, std::memory_order_release);
                // 删除了 ebr.exit_rcu_read();
                return; 
            }
            curr_max_level = max_level_.load(std::memory_order_acquire);
        }

        uint32_t curr_obj = enter_point_id_.load(std::memory_order_acquire);
        float curr_dist = l2_distance_avx2(vector_data, get_node(curr_obj)->vector_data, dim_);

        // 3. 阶段一：找到新节点该插入的目标层的入口（垂直降落）
        // 这里读其他节点的邻居完全无锁，因为没有任何线程会 delete 数组
        for (int level = curr_max_level; level > new_node_level; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                NeighborList* neighbors = get_node(curr_obj)->get_neighbors_rcu(level);
                if (!neighbors) continue;

                for (uint32_t i = 0; i < neighbors->count; ++i) {
                    uint32_t candidate_id = neighbors->neighbors[i];
                    float d = l2_distance_avx2(vector_data, get_node(candidate_id)->vector_data, dim_);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = candidate_id;
                        changed = true;
                    }
                }
            }
        }

        // 4. 阶段二：逐层寻找最近邻并建立双向连接
        int min_level = std::min(curr_max_level, new_node_level);
        for (int level = min_level; level >= 0; --level) {
            // search_layer 内部如果是纯读操作，在 Bulk Load 下也是绝对安全的
            auto top_candidates = search_layer(vector_data, curr_obj, ef_construction_, level);
            
            int num_to_connect = std::min((int)top_candidates.size(), M_);
            for (int i = 0; i < num_to_connect; ++i) {
                uint32_t neighbor_id = top_candidates[i];

                int max_m = (level == 0) ? (M_ * 2) : M_;
                // A. 新节点 -> 邻居 (其实新节点还没暴露，不加锁也没事，但加了逻辑更统一)
                new_node->node_lock.lock();
                this->add_neighbor_inplace(new_node, level, neighbor_id, max_m);
                new_node->node_lock.unlock();
                
                // B. 邻居 -> 新节点 (极大概率被多个线程疯狂竞争，必须上极速自旋锁！)
                HnswNode* neighbor_node = get_node(neighbor_id);
                neighbor_node->node_lock.lock();
                this->add_neighbor_inplace(neighbor_node, level, id, max_m);
                neighbor_node->node_lock.unlock();
            }
            
            if (!top_candidates.empty()) {
                curr_obj = top_candidates[0]; 
            }
        }

        // 5. 阶段三：更新全局最高入口
        if (new_node_level > curr_max_level) {
            std::lock_guard<std::mutex> lock(ep_mutex_); 
            if (new_node_level > max_level_.load(std::memory_order_acquire)) {
                enter_point_id_.store(id, std::memory_order_release);
                max_level_.store(new_node_level, std::memory_order_release);
            }
        }
    }

    // ==========================================
    // 搜索接口 (与你之前的逻辑基本一致，略作精简)
    // ==========================================
    std::vector<uint32_t> search_knn(const float* query, int k, int ef_search) {
        auto& ebr = EBRManager::get_instance();
        ebr.enter_rcu_read();

        int curr_max_level = max_level_.load(std::memory_order_acquire);
        if (curr_max_level == -1) {
            ebr.exit_rcu_read();
            return {};
        }

        uint32_t curr_obj = enter_point_id_.load(std::memory_order_acquire);
        float curr_dist = l2_distance_avx2(query, get_node(curr_obj)->vector_data, dim_);

        for (int level = curr_max_level; level >= 1; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                NeighborList* neighbors = get_node(curr_obj)->get_neighbors_rcu(level);
                if (!neighbors) continue;

                for (uint32_t i = 0; i < neighbors->count; ++i) {
                    uint32_t candidate_id = neighbors->neighbors[i];
                    float d = l2_distance_avx2(query, get_node(candidate_id)->vector_data, dim_);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = candidate_id;
                        changed = true;
                    }
                }
            }
        }

        // 在第 0 层进行精搜
        auto top_k = search_layer(query, curr_obj, std::max(k, ef_search), 0);
        
        ebr.exit_rcu_read();
        
        if (top_k.size() > (size_t)k) top_k.resize(k);
        return top_k;
    }

private:
    size_t dim_;
    size_t max_elements_;
    int M_;
    int ef_construction_;
    double level_mult_;

    HnswNode* nodes_; // 连续内存块首指针

    std::atomic<uint32_t> enter_point_id_;
    std::atomic<int> max_level_;
    std::mutex ep_mutex_; // 仅用于保护极低频的 max_level 更新

    inline void add_neighbor_inplace(HnswNode* node, int layer, uint32_t new_neighbor_id, int max_m) {
        if (layer >= MAX_HNSW_LEVELS) return;

        NeighborList* list = node->neighbor_lists[layer].load(std::memory_order_relaxed); 
        
        // 【防御 1：一次性满额分配，多给 1 个溢出槽位】
        if (list == nullptr) {
            size_t alloc_size = sizeof(NeighborList) + (max_m + 1) * sizeof(uint32_t);
            list = (NeighborList*)std::malloc(alloc_size); 
            list->capacity = max_m + 1;
            list->count = 0;
            node->neighbor_lists[layer].store(list, std::memory_order_release);
        }

        // 去重防御
        for (size_t i = 0; i < list->count; ++i) {
            if (list->neighbors[i] == new_neighbor_id) return;
        }

        // 先无脑塞进去
        list->neighbors[list->count++] = new_neighbor_id;

        // 【核心修复：触发 HNSW 启发式裁剪】
        if (list->count > max_m) {
            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(list->count);
            for (size_t i = 0; i < list->count; ++i) {
                uint32_t cand_id = list->neighbors[i];
                // 现在我们在 Index 内部，直接调用 get_node 和 dim_，没有任何阻碍！
                float dist = l2_distance_avx2(node->vector_data, get_node(cand_id)->vector_data, dim_);
                candidates.push_back({dist, cand_id});
            }

            std::sort(candidates.begin(), candidates.end());

            list->count = 0; 
            for (const auto& cand : candidates) {
                if (list->count >= max_m) break;

                bool keep = true;
                for (size_t i = 0; i < list->count; ++i) {
                    uint32_t selected_id = list->neighbors[i];
                    float dist_to_selected = l2_distance_avx2(
                        get_node(cand.second)->vector_data,
                        get_node(selected_id)->vector_data,
                        dim_
                    );
                    // 启发式：如果离已选邻居更近，则丢弃
                    if (dist_to_selected < cand.first) {
                        keep = false;
                        break;
                    }
                }

                if (keep) {
                    list->neighbors[list->count++] = cand.second;
                }
            }

            // 兜底补齐
            if (list->count < max_m) {
                for (const auto& cand : candidates) {
                    if (list->count >= max_m) break;
                    bool exists = false;
                    for (size_t i = 0; i < list->count; ++i) {
                        if (list->neighbors[i] == cand.second) { exists = true; break; }
                    }
                    if (!exists) {
                        list->neighbors[list->count++] = cand.second;
                    }
                }
            }
        }
    }

    // 随机层数生成器 (多线程安全)
    int get_random_level() {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -std::log(distribution(generator)) * level_mult_;
        return std::min((int)r, MAX_HNSW_LEVELS - 1);
    }

    // 零分配的访问标记集合 (thread_local 完美契合多线程并发建图/查询)
    bool is_visited(uint32_t id) {
        static thread_local std::vector<uint32_t> visited_array;
        static thread_local uint32_t current_version = 0;
        
        if (visited_array.size() <= id) visited_array.resize(max_elements_, 0);
        
        if (id == 0xFFFFFFFF) { // 特殊标记：重置版本号
           current_version++;
           if (current_version == 0) { 
               std::fill(visited_array.begin(), visited_array.end(), 0);
               current_version = 1;
           }
           return true;
        }

        if (visited_array[id] == current_version) return true;
        visited_array[id] = current_version;
        return false;
    }

    // 通用的单层启发式搜索
    std::vector<uint32_t> search_layer(const float* query, uint32_t ep_id, int ef, int level) {
        std::priority_queue<NodeDist> top_candidates;
        std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> candidates;

        float ep_dist = l2_distance_avx2(query, get_node(ep_id)->vector_data, dim_);
        
        is_visited(0xFFFFFFFF); // 重置 visited
        is_visited(ep_id);

        candidates.push({ep_id, ep_dist});
        top_candidates.push({ep_id, ep_dist});

        while (!candidates.empty()) {
            NodeDist current = candidates.top();
            candidates.pop();

            if (current.dist > top_candidates.top().dist && top_candidates.size() == (size_t)ef) {
                break; 
            }

            NeighborList* neighbors = get_node(current.id)->get_neighbors_rcu(level);
            if (!neighbors) continue;

            for (uint32_t i = 0; i < neighbors->count; ++i) {
                uint32_t neighbor_id = neighbors->neighbors[i];
                if (!is_visited(neighbor_id)) {
                    float d = l2_distance_avx2(query, get_node(neighbor_id)->vector_data, dim_);
                    
                    if (top_candidates.size() < (size_t)ef || d < top_candidates.top().dist) {
                        candidates.push({neighbor_id, d});
                        top_candidates.push({neighbor_id, d});
                        if (top_candidates.size() > (size_t)ef) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        std::vector<uint32_t> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            result.push_back(top_candidates.top().id);
            top_candidates.pop();
        }
        std::reverse(result.begin(), result.end()); // 返回由近到远的有序数组
        return result;
    }
};

} // namespace vector_search