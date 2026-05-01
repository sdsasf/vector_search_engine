#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#include <immintrin.h>

#include "hnsw_index.h"
#include "persistence.h"
#include "write_buffer.h"

namespace vector_search {

class VectorEngine {
public:
    VectorEngine(size_t dim, size_t max_elements, int M = 16, int ef_construction = 200,
                 size_t buffer_cap = 50000, int bg_threads = 2,
                 const std::string& persistence_dir = "")
        : dim_(dim), buffer_capacity_(buffer_cap), hnsw_index_(nullptr),
          soft_limit_(3), hard_limit_(6), running_(true), pending_flushes_(0),
          persistence_dir_(persistence_dir), loaded_snapshot_(false) {
        hnsw_index_ = new HnswIndex(dim, max_elements, M, ef_construction);
        active_buffer_ = std::make_shared<FlatWriteBuffer>(buffer_capacity_, dim_);

        if (!persistence_dir_.empty()) {
            std::filesystem::create_directories(persistence_dir_);
            snapshot_path_ = (std::filesystem::path(persistence_dir_) / "snapshot.bin").string();
            wal_path_ = (std::filesystem::path(persistence_dir_) / "wal.log").string();

            uint64_t last_lsn = 0;
            if (std::filesystem::exists(snapshot_path_)) {
                loaded_snapshot_ = hnsw_index_->load_snapshot(snapshot_path_);
                last_lsn = replay_wal(wal_path_, hnsw_index_->snapshot_lsn(), static_cast<uint32_t>(dim_),
                                      [this](uint32_t id, const float* vec) {
                                          hnsw_index_->insert_bulk(vec, id);
                                      });
            }

            wal_ = std::make_unique<WalWriter>(wal_path_, last_lsn);
            if (!loaded_snapshot_) {
                wal_->reset(1);
            }
        }

        start_background_threads(bg_threads);
    }

    ~VectorEngine() {
        try {
            if (wal_) {
                save_snapshot_and_reset_wal();
            }
        } catch (...) {
        }

        running_.store(false);
        bg_cv_.notify_all();
        for (auto& t : bg_flush_threads_) {
            if (t.joinable()) t.join();
        }
        delete hnsw_index_;
    }

    HnswIndex* get_raw_index() { return hnsw_index_; }

    bool has_persisted_data() const { return loaded_snapshot_; }
    bool persistence_enabled() const { return wal_ != nullptr; }

    bool save_snapshot_and_reset_wal() {
        if (!wal_) return true;
        drain_buffers_for_snapshot();
        const uint64_t lsn = wal_ ? wal_->current_lsn() : hnsw_index_->snapshot_lsn();
        if (!hnsw_index_->save_snapshot(snapshot_path_, lsn)) {
            return false;
        }
        if (wal_) {
            wal_->reset(lsn + 1);
        }
        loaded_snapshot_ = true;
        return true;
    }

    void insert(const float* vec, uint32_t id) {
        if (wal_) {
            wal_->append_insert(id, vec, static_cast<uint32_t>(dim_));
        }

        if (active_buffer_->append_wait_free(vec, id)) return;

        std::unique_lock<std::mutex> lock(swap_mutex_);
        if (active_buffer_->append_wait_free(vec, id)) return;

        size_t q_size = immutable_queue_.size();
        if (q_size >= soft_limit_ && q_size < hard_limit_) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            lock.lock();
        }

        swap_cv_.wait(lock, [this]() { return immutable_queue_.size() < hard_limit_; });

        immutable_queue_.push(active_buffer_);
        active_buffer_ = std::make_shared<FlatWriteBuffer>(buffer_capacity_, dim_);
        active_buffer_->append_wait_free(vec, id);
        bg_cv_.notify_one();
    }

    std::vector<uint32_t> search_knn(const float* query, int k, int ef_search) {
        std::priority_queue<NodeDist> top_candidates;

        std::vector<std::shared_ptr<FlatWriteBuffer>> imm_snapshots;
        std::shared_ptr<FlatWriteBuffer> active_snap;
        {
            std::lock_guard<std::mutex> lock(swap_mutex_);
            active_snap = active_buffer_;
            auto q_copy = immutable_queue_;
            while (!q_copy.empty()) {
                imm_snapshots.push_back(q_copy.front());
                q_copy.pop();
            }
        }

        for (auto& imm_ptr : imm_snapshots) {
            imm_ptr->search_brute_force(query, k, top_candidates);
        }
        active_snap->search_brute_force(query, k, top_candidates);

        auto hnsw_results = hnsw_index_->search_knn(query, k, ef_search);
        for (uint32_t id : hnsw_results) {
            float d = l2_distance_avx2(query, hnsw_index_->get_node(id)->vector_data, dim_);
            if (top_candidates.size() < static_cast<size_t>(k) || d < top_candidates.top().dist) {
                top_candidates.push({id, d});
                if (top_candidates.size() > static_cast<size_t>(k)) top_candidates.pop();
            }
        }

        std::vector<uint32_t> result;
        while (!top_candidates.empty()) {
            result.push_back(top_candidates.top().id);
            top_candidates.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

private:
    void start_background_threads(int bg_threads) {
        int num_cores = std::thread::hardware_concurrency();
        for (int i = 0; i < bg_threads; ++i) {
            bg_flush_threads_.emplace_back(&VectorEngine::background_flush_loop, this);
            if (num_cores >= 4) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                int target_core = num_cores - 1 - (i % 2);
                CPU_SET(target_core, &cpuset);
                (void)pthread_setaffinity_np(bg_flush_threads_.back().native_handle(), sizeof(cpu_set_t), &cpuset);
            }
        }
    }

    void flush_buffer_to_index(const std::shared_ptr<FlatWriteBuffer>& buffer) {
        size_t count = buffer->visible_count();
        for (size_t i = 0; i < count; ++i) {
            while (!buffer->is_ready(i)) {
                _mm_pause();
            }
            hnsw_index_->insert(buffer->data + i * dim_, buffer->ids[i]);
        }
    }

    void background_flush_loop() {
        while (true) {
            std::shared_ptr<FlatWriteBuffer> buffer_to_flush;
            {
                std::unique_lock<std::mutex> lock(swap_mutex_);
                bg_cv_.wait(lock, [this]() { return !immutable_queue_.empty() || !running_.load(); });
                if (!running_.load() && immutable_queue_.empty()) break;

                buffer_to_flush = immutable_queue_.front();
                immutable_queue_.pop();
                pending_flushes_.fetch_add(1, std::memory_order_acq_rel);
            }

            flush_buffer_to_index(buffer_to_flush);

            pending_flushes_.fetch_sub(1, std::memory_order_acq_rel);
            flush_cv_.notify_all();
            swap_cv_.notify_all();
        }
    }

    void drain_buffers_for_snapshot() {
        std::vector<std::shared_ptr<FlatWriteBuffer>> buffers;
        {
            std::lock_guard<std::mutex> lock(swap_mutex_);
            while (!immutable_queue_.empty()) {
                buffers.push_back(immutable_queue_.front());
                immutable_queue_.pop();
            }
            if (active_buffer_ && active_buffer_->visible_count() > 0) {
                buffers.push_back(active_buffer_);
                active_buffer_ = std::make_shared<FlatWriteBuffer>(buffer_capacity_, dim_);
            }
        }

        for (auto& buffer : buffers) {
            flush_buffer_to_index(buffer);
        }

        std::unique_lock<std::mutex> lock(swap_mutex_);
        flush_cv_.wait(lock, [this]() {
            return immutable_queue_.empty() && pending_flushes_.load(std::memory_order_acquire) == 0;
        });
    }

    size_t dim_;
    size_t buffer_capacity_;
    HnswIndex* hnsw_index_;

    std::shared_ptr<FlatWriteBuffer> active_buffer_;
    std::queue<std::shared_ptr<FlatWriteBuffer>> immutable_queue_;

    size_t soft_limit_;
    size_t hard_limit_;

    std::mutex swap_mutex_;
    std::condition_variable swap_cv_;
    std::condition_variable flush_cv_;

    std::vector<std::thread> bg_flush_threads_;
    std::condition_variable bg_cv_;
    std::atomic<bool> running_;
    std::atomic<size_t> pending_flushes_;

    std::string persistence_dir_;
    std::string snapshot_path_;
    std::string wal_path_;
    std::unique_ptr<WalWriter> wal_;
    bool loaded_snapshot_;
};

} // namespace vector_search
