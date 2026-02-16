#pragma once
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>

namespace vector_search {

class EBRManager {
public:
    // 获取单例
    static EBRManager& get_instance() {
        static EBRManager instance;
        return instance;
    }

    // 【读线程专用】进入 RCU 临界区（开始搜索）
    void enter_rcu_read() {
        // 读取当前的全局纪元，并记录到当前线程的局部变量中
        uint64_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        local_epochs_[get_thread_id()].val.store(current_epoch, std::memory_order_release);
        active_threads_[get_thread_id()].val.store(true, std::memory_order_release);
    }

    // 【读线程专用】离开 RCU 临界区（搜索结束）
    void exit_rcu_read() {
        active_threads_[get_thread_id()].val.store(false, std::memory_order_release);
    }

    // 【写线程专用】把废弃的内存扔进延迟回收队列
    void defer_free(void* ptr) {
        uint64_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        
        std::lock_guard<std::mutex> lock(retire_mutex_);
        retire_list_.push_back({ptr, current_epoch});
        
        // 每积攒一定数量的废弃内存，尝试推进一次纪元并清理
        if (retire_list_.size() >= 1024) {
            try_reclaim();
        }
    }

private:
    EBRManager() {}

    // 推进纪元并回收安全内存
    void try_reclaim() {
        uint64_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        
        // 检查所有活跃线程，找出它们中最老的那个纪元
        uint64_t min_active_epoch = current_epoch;
        for (size_t i = 0; i < MAX_THREADS; ++i) {
            if (active_threads_[i].val.load(std::memory_order_acquire)) {
                uint64_t thread_epoch = local_epochs_[i].val.load(std::memory_order_acquire);
                if (thread_epoch < min_active_epoch) {
                    min_active_epoch = thread_epoch;
                }
            }
        }

        // 如果所有活跃线程都跟上了当前纪元，我们就可以将全局纪元 +1
        if (min_active_epoch == current_epoch) {
            global_epoch_.fetch_add(1, std::memory_order_release);
        }

        // 回收那些在 min_active_epoch 之前被废弃的内存
        // 因为已经没有任何活跃线程能访问到它们了！
        auto it = retire_list_.begin();
        while (it != retire_list_.end()) {
            if (it->retire_epoch < min_active_epoch) {
                free(it->ptr); // 真正释放物理内存！
                it = retire_list_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 辅助函数：分配一个简单的线程 ID (0 ~ MAX_THREADS-1)
    static size_t get_thread_id() {
        static std::atomic<size_t> id_counter{0};
        thread_local size_t thread_id = id_counter.fetch_add(1);
        return thread_id;
    }

    // 定义常量和状态
    static constexpr size_t MAX_THREADS = 128;
    
    std::atomic<uint64_t> global_epoch_{1};
    
    // 用数组来避免 False Sharing 的问题（实际工程中这里需要 alignas(64) 对齐）
    struct alignas(64) ThreadState { std::atomic<uint64_t> val{0}; };
    struct alignas(64) ThreadActive { std::atomic<bool> val{false}; };
    
    ThreadState local_epochs_[MAX_THREADS];
    ThreadActive active_threads_[MAX_THREADS];

    // 废弃内存块记录
    struct RetiredPtr {
        void* ptr;
        uint64_t retire_epoch;
    };
    
    std::mutex retire_mutex_; // 保护垃圾桶的锁（写线程本来就不多，这里用锁是可以接受的）
    std::vector<RetiredPtr> retire_list_;
};

} // namespace vector_search