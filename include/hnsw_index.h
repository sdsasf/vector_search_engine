#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>
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
    HnswIndex(size_t dim, size_t max_elements, int M = 16, int ef_construction = 100)
        : dim_(dim), max_elements_(max_elements), M_(M), ef_construction_(ef_construction) {
        nodes_ = static_cast<HnswNode*>(::operator new[](max_elements_ * sizeof(HnswNode), std::align_val_t(CACHE_LINE_SIZE)));
        for (size_t i = 0; i < max_elements_; ++i) {
            new (&nodes_[i]) HnswNode();
        }

        const size_t vector_bytes = max_elements_ * dim_ * sizeof(float);
        const size_t aligned_vector_bytes = ((vector_bytes + 31) / 32) * 32;
        vector_storage_ = static_cast<float*>(std::aligned_alloc(32, aligned_vector_bytes));
        if (vector_storage_ == nullptr) {
            throw std::bad_alloc();
        }

        active_flags_ = new std::atomic<uint8_t>[max_elements_];
        for (size_t i = 0; i < max_elements_; ++i) {
            active_flags_[i].store(0, std::memory_order_relaxed);
        }

        level_mult_ = 1.0 / std::log(1.0 * M_);
        enter_point_id_.store(0, std::memory_order_relaxed);
        max_level_.store(-1, std::memory_order_relaxed);
        element_count_.store(0, std::memory_order_relaxed);
        max_id_seen_.store(0, std::memory_order_relaxed);
        snapshot_lsn_.store(0, std::memory_order_relaxed);
    }

    ~HnswIndex() {
        for (size_t id = 0; id < max_elements_; ++id) {
            if (active_flags_[id].load(std::memory_order_relaxed) == 0) continue;
            for (int level = 0; level < MAX_HNSW_LEVELS; ++level) {
                NeighborList* list = nodes_[id].neighbor_lists[level].load(std::memory_order_relaxed);
                std::free(list);
            }
        }
        delete[] active_flags_;
        std::free(vector_storage_);
        for (size_t i = 0; i < max_elements_; ++i) {
            nodes_[i].~HnswNode();
        }
        ::operator delete[](nodes_, std::align_val_t(CACHE_LINE_SIZE));
    }

    inline HnswNode* get_node(uint32_t id) { return &nodes_[id]; }
    inline const HnswNode* get_node(uint32_t id) const { return &nodes_[id]; }

    size_t element_count() const { return element_count_.load(std::memory_order_acquire); }
    uint64_t snapshot_lsn() const { return snapshot_lsn_.load(std::memory_order_acquire); }

    void insert(const float* vector_data, uint32_t id) {
        auto& ebr = EBRManager::get_instance();
        ebr.enter_rcu_read();
        insert_internal(vector_data, id, false);
        ebr.exit_rcu_read();
    }

    void insert_bulk(const float* vector_data, uint32_t id) {
        insert_internal(vector_data, id, true);
    }

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

        auto top_k = search_layer(query, curr_obj, std::max(k, ef_search), 0);
        ebr.exit_rcu_read();

        if (top_k.size() > static_cast<size_t>(k)) top_k.resize(k);
        return top_k;
    }

    bool save_snapshot(const std::string& path, uint64_t lsn) {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        const std::string tmp_path = path + ".tmp";
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) return false;

        SnapshotHeader header{};
        std::memcpy(header.magic, kSnapshotMagic, sizeof(header.magic));
        header.version = kSnapshotVersion;
        header.dim = static_cast<uint32_t>(dim_);
        header.max_elements = static_cast<uint32_t>(max_elements_);
        header.M = static_cast<uint32_t>(M_);
        header.ef_construction = static_cast<uint32_t>(ef_construction_);
        header.max_level = max_level_.load(std::memory_order_acquire);
        header.enter_point_id = enter_point_id_.load(std::memory_order_acquire);
        header.element_count = element_count_.load(std::memory_order_acquire);
        header.max_id_seen = max_id_seen_.load(std::memory_order_acquire);
        header.lsn = lsn;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        const uint32_t max_id = header.max_id_seen;
        for (uint32_t id = 0; id <= max_id && id < max_elements_; ++id) {
            if (active_flags_[id].load(std::memory_order_acquire) == 0) continue;
            const HnswNode* node = get_node(id);
            PersistedNodeHeader node_header{id, node->level};
            out.write(reinterpret_cast<const char*>(&node_header), sizeof(node_header));
            out.write(reinterpret_cast<const char*>(node->vector_data), dim_ * sizeof(float));

            for (int level = 0; level <= node->level && level < MAX_HNSW_LEVELS; ++level) {
                NeighborList* list = node->get_neighbors_rcu(level);
                uint32_t count = list ? list->count : 0;
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                if (count > 0) {
                    out.write(reinterpret_cast<const char*>(list->neighbors), count * sizeof(uint32_t));
                }
            }
        }

        if (!out) return false;
        out.close();
        std::filesystem::rename(tmp_path, path);
        snapshot_lsn_.store(lsn, std::memory_order_release);
        return true;
    }

    bool load_snapshot(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        SnapshotHeader header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in || std::memcmp(header.magic, kSnapshotMagic, sizeof(header.magic)) != 0 ||
            header.version != kSnapshotVersion || header.dim != dim_ || header.max_elements > max_elements_ ||
            header.M != static_cast<uint32_t>(M_) || header.element_count > max_elements_) {
            throw std::runtime_error("Unsupported or corrupt snapshot: " + path);
        }

        for (uint64_t i = 0; i < header.element_count; ++i) {
            PersistedNodeHeader node_header{};
            in.read(reinterpret_cast<char*>(&node_header), sizeof(node_header));
            if (!in || node_header.id >= max_elements_ || node_header.level < 0 || node_header.level >= MAX_HNSW_LEVELS) {
                throw std::runtime_error("Corrupt snapshot node: " + path);
            }

            float* stored_vector = vector_storage_ + node_header.id * dim_;
            in.read(reinterpret_cast<char*>(stored_vector), dim_ * sizeof(float));
            if (!in) throw std::runtime_error("Corrupt snapshot vector: " + path);

            HnswNode* node = get_node(node_header.id);
            node->init(stored_vector, node_header.level);
            for (int level = 0; level <= node_header.level; ++level) {
                uint32_t count = 0;
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                if (!in) throw std::runtime_error("Corrupt snapshot adjacency: " + path);

                size_t alloc_size = sizeof(NeighborList) + count * sizeof(uint32_t);
                NeighborList* list = static_cast<NeighborList*>(std::malloc(alloc_size));
                if (list == nullptr) throw std::bad_alloc();
                list->capacity = count;
                list->count = count;
                if (count > 0) {
                    in.read(reinterpret_cast<char*>(list->neighbors), count * sizeof(uint32_t));
                    if (!in) throw std::runtime_error("Corrupt snapshot neighbors: " + path);
                }
                node->neighbor_lists[level].store(list, std::memory_order_release);
            }
            mark_active(node_header.id);
            update_max_id_seen(node_header.id);
        }

        max_level_.store(header.max_level, std::memory_order_release);
        enter_point_id_.store(header.enter_point_id, std::memory_order_release);
        snapshot_lsn_.store(header.lsn, std::memory_order_release);
        return true;
    }

private:
    size_t dim_;
    size_t max_elements_;
    int M_;
    int ef_construction_;
    double level_mult_;

    HnswNode* nodes_;
    float* vector_storage_;
    std::atomic<uint8_t>* active_flags_;

    std::atomic<uint32_t> enter_point_id_;
    std::atomic<int> max_level_;
    std::atomic<size_t> element_count_;
    std::atomic<uint32_t> max_id_seen_;
    std::atomic<uint64_t> snapshot_lsn_;
    std::mutex ep_mutex_;

    static constexpr char kSnapshotMagic[8] = {'V', 'S', 'E', 'S', 'N', 'A', 'P', '1'};
    static constexpr uint32_t kSnapshotVersion = 1;

    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t max_elements;
        uint32_t M;
        uint32_t ef_construction;
        int32_t max_level;
        uint32_t enter_point_id;
        uint64_t element_count;
        uint32_t max_id_seen;
        uint64_t lsn;
    };

    struct PersistedNodeHeader {
        uint32_t id;
        int32_t level;
    };

    void insert_internal(const float* vector_data, uint32_t id, bool bulk_mode) {
        if (id >= max_elements_) {
            throw std::out_of_range("HNSW id exceeds max_elements");
        }

        int new_node_level = get_random_level();
        HnswNode* new_node = get_node(id);
        float* stored_vector = vector_storage_ + id * dim_;
        std::memcpy(stored_vector, vector_data, dim_ * sizeof(float));
        new_node->init(stored_vector, new_node_level);
        update_max_id_seen(id);

        int curr_max_level = max_level_.load(std::memory_order_acquire);
        if (curr_max_level == -1) {
            std::lock_guard<std::mutex> lock(ep_mutex_);
            if (max_level_.load(std::memory_order_acquire) == -1) {
                enter_point_id_.store(id, std::memory_order_release);
                max_level_.store(new_node_level, std::memory_order_release);
                mark_active(id);
                return;
            }
            curr_max_level = max_level_.load(std::memory_order_acquire);
        }

        uint32_t curr_obj = enter_point_id_.load(std::memory_order_acquire);
        float curr_dist = l2_distance_avx2(stored_vector, get_node(curr_obj)->vector_data, dim_);

        for (int level = curr_max_level; level > new_node_level; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                NeighborList* neighbors = get_node(curr_obj)->get_neighbors_rcu(level);
                if (!neighbors) continue;

                for (uint32_t i = 0; i < neighbors->count; ++i) {
                    uint32_t candidate_id = neighbors->neighbors[i];
                    float d = l2_distance_avx2(stored_vector, get_node(candidate_id)->vector_data, dim_);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = candidate_id;
                        changed = true;
                    }
                }
            }
        }

        int min_level = std::min(curr_max_level, new_node_level);
        for (int level = min_level; level >= 0; --level) {
            auto top_candidates = search_layer(stored_vector, curr_obj, ef_construction_, level);
            int num_to_connect = std::min(static_cast<int>(top_candidates.size()), M_);

            for (int i = 0; i < num_to_connect; ++i) {
                uint32_t neighbor_id = top_candidates[i];
                if (bulk_mode) {
                    int max_m = (level == 0) ? (M_ * 2) : M_;
                    new_node->node_lock.lock();
                    add_neighbor_inplace(new_node, level, neighbor_id, max_m);
                    new_node->node_lock.unlock();

                    HnswNode* neighbor_node = get_node(neighbor_id);
                    neighbor_node->node_lock.lock();
                    add_neighbor_inplace(neighbor_node, level, id, max_m);
                    neighbor_node->node_lock.unlock();
                } else {
                    new_node->add_neighbor_rcu(level, neighbor_id);
                    get_node(neighbor_id)->add_neighbor_rcu(level, id);
                }
            }

            if (!top_candidates.empty()) {
                curr_obj = top_candidates[0];
            }
        }

        if (new_node_level > curr_max_level) {
            std::lock_guard<std::mutex> lock(ep_mutex_);
            if (new_node_level > max_level_.load(std::memory_order_acquire)) {
                enter_point_id_.store(id, std::memory_order_release);
                max_level_.store(new_node_level, std::memory_order_release);
            }
        }

        mark_active(id);
    }

    inline void add_neighbor_inplace(HnswNode* node, int layer, uint32_t new_neighbor_id, int max_m) {
        if (layer >= MAX_HNSW_LEVELS) return;

        NeighborList* list = node->neighbor_lists[layer].load(std::memory_order_relaxed);
        if (list == nullptr) {
            size_t alloc_size = sizeof(NeighborList) + (max_m + 1) * sizeof(uint32_t);
            list = static_cast<NeighborList*>(std::malloc(alloc_size));
            if (list == nullptr) throw std::bad_alloc();
            list->capacity = max_m + 1;
            list->count = 0;
            node->neighbor_lists[layer].store(list, std::memory_order_release);
        }

        for (size_t i = 0; i < list->count; ++i) {
            if (list->neighbors[i] == new_neighbor_id) return;
        }

        list->neighbors[list->count++] = new_neighbor_id;
        if (list->count <= static_cast<uint32_t>(max_m)) return;

        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(list->count);
        for (size_t i = 0; i < list->count; ++i) {
            uint32_t cand_id = list->neighbors[i];
            float dist = l2_distance_avx2(node->vector_data, get_node(cand_id)->vector_data, dim_);
            candidates.push_back({dist, cand_id});
        }

        std::sort(candidates.begin(), candidates.end());
        list->count = 0;
        for (const auto& cand : candidates) {
            if (list->count >= static_cast<uint32_t>(max_m)) break;

            bool keep = true;
            for (size_t i = 0; i < list->count; ++i) {
                uint32_t selected_id = list->neighbors[i];
                float dist_to_selected = l2_distance_avx2(
                    get_node(cand.second)->vector_data,
                    get_node(selected_id)->vector_data,
                    dim_);
                if (dist_to_selected < cand.first) {
                    keep = false;
                    break;
                }
            }
            if (keep) list->neighbors[list->count++] = cand.second;
        }

        if (list->count < static_cast<uint32_t>(max_m)) {
            for (const auto& cand : candidates) {
                if (list->count >= static_cast<uint32_t>(max_m)) break;
                bool exists = false;
                for (size_t i = 0; i < list->count; ++i) {
                    if (list->neighbors[i] == cand.second) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) list->neighbors[list->count++] = cand.second;
            }
        }
    }

    int get_random_level() {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -std::log(distribution(generator)) * level_mult_;
        return std::min(static_cast<int>(r), MAX_HNSW_LEVELS - 1);
    }

    bool is_visited(uint32_t id) {
        static thread_local std::vector<uint32_t> visited_array;
        static thread_local uint32_t current_version = 0;

        if (visited_array.size() <= id) visited_array.resize(max_elements_, 0);

        if (id == 0xFFFFFFFF) {
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

    std::vector<uint32_t> search_layer(const float* query, uint32_t ep_id, int ef, int level) {
        std::priority_queue<NodeDist> top_candidates;
        std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> candidates;

        float ep_dist = l2_distance_avx2(query, get_node(ep_id)->vector_data, dim_);
        is_visited(0xFFFFFFFF);
        is_visited(ep_id);

        candidates.push({ep_id, ep_dist});
        top_candidates.push({ep_id, ep_dist});

        while (!candidates.empty()) {
            NodeDist current = candidates.top();
            candidates.pop();

            if (current.dist > top_candidates.top().dist && top_candidates.size() == static_cast<size_t>(ef)) {
                break;
            }

            NeighborList* neighbors = get_node(current.id)->get_neighbors_rcu(level);
            if (!neighbors) continue;

            for (uint32_t i = 0; i < neighbors->count; ++i) {
                uint32_t neighbor_id = neighbors->neighbors[i];
                if (!is_visited(neighbor_id)) {
                    float d = l2_distance_avx2(query, get_node(neighbor_id)->vector_data, dim_);
                    if (top_candidates.size() < static_cast<size_t>(ef) || d < top_candidates.top().dist) {
                        candidates.push({neighbor_id, d});
                        top_candidates.push({neighbor_id, d});
                        if (top_candidates.size() > static_cast<size_t>(ef)) {
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
        std::reverse(result.begin(), result.end());
        return result;
    }

    void mark_active(uint32_t id) {
        uint8_t expected = 0;
        if (active_flags_[id].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            element_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void update_max_id_seen(uint32_t id) {
        uint32_t current = max_id_seen_.load(std::memory_order_relaxed);
        while (id > current && !max_id_seen_.compare_exchange_weak(
                   current, id, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }
};

} // namespace vector_search
