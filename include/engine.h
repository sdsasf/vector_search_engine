#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <pthread.h>
#include <atomic>
#include <queue>
#include <vector>
#include <memory>
#include "hnsw_index.h"
#include "write_buffer.h"

namespace vector_search {

class VectorEngine {
public:
    // 参数升级：增加后台线程数 (bg_threads)、软限制 (soft_limit)、硬限制 (hard_limit)
    VectorEngine(size_t dim, size_t max_elements, int M = 16, int ef_construction = 200, 
                 size_t buffer_cap = 50000, int bg_threads = 2)
        : dim_(dim), buffer_capacity_(buffer_cap), running_(true),
          soft_limit_(3), hard_limit_(6) { // 堆积3个开始降速，堆积6个开始死等
        
        hnsw_index_ = new HnswIndex(dim, max_elements, M, ef_construction);
        
        // 使用 shared_ptr 管理 Active Buffer，完美解决多线程生命周期问题
        active_buffer_ = std::make_shared<FlatWriteBuffer>(buffer_capacity_, dim_);
        
        // 招式三：并发 Compaction（启动多线程后台建图池）
        int num_cores = std::thread::hardware_concurrency();
        
        for (int i = 0; i < bg_threads; ++i) {
            bg_flush_threads_.emplace_back(&VectorEngine::background_flush_loop, this);
            
            // 【硬核绑核逻辑】：只在核心数充裕时进行隔离
            if (num_cores >= 4) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                
                // 策略：把后台建图线程死死钉在最后两个核心上 
                // 比如 num_cores=6，那么 i=0绑核5，i=1绑核4，i=2绑核5...
                int target_core = num_cores - 1 - (i % 2); 
                CPU_SET(target_core, &cpuset);
                
                // 调用 Linux 原生 API 设置亲和性
                int rc = pthread_setaffinity_np(bg_flush_threads_.back().native_handle(),
                                                sizeof(cpu_set_t), &cpuset);
                if (rc != 0) {
                    // 如果系统权限不够导致绑核失败，静默忽略即可
                }
            }
        }
    }

    ~VectorEngine() {
        running_.store(false);
        bg_cv_.notify_all(); // 唤醒所有后台线程退出
        for (auto& t : bg_flush_threads_) {
            if (t.joinable()) t.join();
        }
        delete hnsw_index_;
    }

    // 暴露底层的 HNSW 索引，专供 Server 启动时的全量并发导入 (Bulk Load) 使用
    HnswIndex* get_raw_index() { return hnsw_index_; }

    // 【前台写入：融合背压与无锁队列】
    void insert(const float* vec, uint32_t id) {
        if (active_buffer_->append_wait_free(vec, id)) return;

        std::unique_lock<std::mutex> lock(swap_mutex_);
        if (active_buffer_->append_wait_free(vec, id)) return;

        size_t q_size = immutable_queue_.size();

        // 招式二：Write Throttling (平滑背压限流)
        if (q_size >= soft_limit_ && q_size < hard_limit_) {
            // 不阻塞，但让线程微睡眠，强行降低前台写入的 QPS，保护内存
            lock.unlock(); // 睡眠前释放锁
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            lock.lock();
        }

        // 招式一：Immutable 队列的终极硬限流 (防止 OOM)
        swap_cv_.wait(lock, [this]() { 
            return immutable_queue_.size() < hard_limit_; 
        });

        // 将满的 Buffer 推入队列
        immutable_queue_.push(active_buffer_);
        
        // 瞬间分配新的 Active Buffer 接客
        active_buffer_ = std::make_shared<FlatWriteBuffer>(buffer_capacity_, dim_);
        active_buffer_->append_wait_free(vec, id);

        // 唤醒一个空闲的后台线程去干活
        bg_cv_.notify_one(); 
    }

    // 【前台搜索：极其安全的快照多路归并】
    std::vector<uint32_t> search_knn(const float* query, int k, int ef_search) {
        std::priority_queue<NodeDist> top_candidates;

        // 极其轻量级的快照拷贝：利用 shared_ptr，即使后台线程弹出了队列并销毁它，
        // 只要这个快照 vector 里还有它的 shared_ptr，这块内存就绝对安全！
        std::vector<std::shared_ptr<FlatWriteBuffer>> imm_snapshots;
        std::shared_ptr<FlatWriteBuffer> active_snap;
        {
            std::lock_guard<std::mutex> lock(swap_mutex_);
            active_snap = active_buffer_;
            
            // 拷贝底层容器
            auto q_copy = immutable_queue_;
            while(!q_copy.empty()) {
                imm_snapshots.push_back(q_copy.front());
                q_copy.pop();
            }
        }

        // 1. 暴力搜所有的 Immutable Buffer
        for (auto& imm_ptr : imm_snapshots) {
            imm_ptr->search_brute_force(query, k, top_candidates);
        }

        // 2. 暴力搜 Active Buffer
        active_snap->search_brute_force(query, k, top_candidates);

        // 3. 搜底层的静态 HNSW 图
        auto hnsw_results = hnsw_index_->search_knn(query, k, ef_search);
        for (uint32_t id : hnsw_results) {
            float d = l2_distance_avx2(query, hnsw_index_->get_node(id)->vector_data, dim_);
            if (top_candidates.size() < (size_t)k || d < top_candidates.top().dist) {
                top_candidates.push({id, d});
                if (top_candidates.size() > (size_t)k) top_candidates.pop();
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
    void background_flush_loop() {
        while (running_.load()) {
            std::shared_ptr<FlatWriteBuffer> buffer_to_flush;
            {
                std::unique_lock<std::mutex> lock(swap_mutex_);
                bg_cv_.wait(lock, [this]() { return !immutable_queue_.empty() || !running_.load(); });
                if (!running_.load() && immutable_queue_.empty()) break;
                
                buffer_to_flush = immutable_queue_.front();
                immutable_queue_.pop();
            }

            // 脱离锁：后台并发疯狂建图 (HNSW 内部受 RCU 保护，天生支持多线程建图)
            size_t count = buffer_to_flush->count.load(std::memory_order_acquire);
            if (count > buffer_capacity_) count = buffer_capacity_;
            
            for (size_t i = 0; i < count; ++i) {
                hnsw_index_->insert(buffer_to_flush->data + i * dim_, buffer_to_flush->ids[i]);
            }

            {
                std::lock_guard<std::mutex> lock(swap_mutex_);
                archive_buffers_.push_back(buffer_to_flush);
            }
            // 刷盘完毕！
            // buffer_to_flush 离开作用域，shared_ptr 计数减 1。
            // 如果没有读线程在使用它，它将自动触发析构函数释放内存。
            swap_cv_.notify_all(); // 通知前台，队列腾出空间了
        }
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
    
    std::vector<std::thread> bg_flush_threads_;
    std::condition_variable bg_cv_;
    std::atomic<bool> running_;

    std::vector<std::shared_ptr<FlatWriteBuffer>> archive_buffers_;
};

} // namespace vector_search