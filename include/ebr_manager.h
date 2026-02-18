#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <vector>

namespace vector_search {

// 高性能 EBR（Epoch-Based Reclamation）管理器。
// 设计目标：
// 1) 读侧开销极低（pin/unpin 仅 thread-local + 原子读写）；
// 2) 写侧延迟回收（本地批量退休 + 全局分桶回收）；
// 3) 支持线程动态注册/注销；
// 4) C++17 可编译，支持自定义 deleter。
class EBRManager {
public:
    static EBRManager& get_instance() {
        static EBRManager instance;
        return instance;
    }

    // 进入读侧临界区（支持嵌套）
    void enter_rcu_read() {
        Participant& p = local_participant();
        const uint32_t prev = p.pin_count.load(std::memory_order_relaxed);
        if (prev == 0U) {
            const uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
            p.local_epoch.store(epoch, std::memory_order_release);
            p.active.store(true, std::memory_order_release);
        }
        p.pin_count.store(prev + 1U, std::memory_order_relaxed);
    }

    // 离开读侧临界区
    void exit_rcu_read() {
        Participant& p = local_participant();
        const uint32_t prev = p.pin_count.load(std::memory_order_relaxed);
        if (prev <= 1U) {
            p.pin_count.store(0U, std::memory_order_relaxed);
            p.active.store(false, std::memory_order_release);
            maybe_flush_local_retired(p);
            return;
        }
        p.pin_count.store(prev - 1U, std::memory_order_relaxed);
    }

    // 延迟释放 malloc/new[] 等兼容 free 的内存。
    void defer_free(void* ptr) {
        defer_delete(ptr, &EBRManager::free_deleter);
    }

    // 泛型延迟删除，默认使用 delete。
    template <typename T>
    void defer_delete(T* ptr) {
        static_assert(!std::is_void<T>::value, "void* please use defer_free or custom deleter");
        defer_delete(static_cast<void*>(ptr), &EBRManager::typed_delete<T>);
    }

    // 延迟删除（自定义 deleter）。
    void defer_delete(void* ptr, void (*deleter)(void*)) {
        if (ptr == nullptr || deleter == nullptr) {
            return;
        }

        Participant& p = local_participant();
        const uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
        p.local_retired.emplace_back(RetiredNode{ptr, deleter, epoch});

        if (p.local_retired.size() >= kLocalBatchThreshold) {
            flush_local_retired(p);
            try_advance_epoch_and_reclaim();
        }
    }

    // 主动触发一次回收尝试（可在后台线程周期调用）
    void collect() {
        Participant& p = local_participant();
        flush_local_retired(p);
        try_advance_epoch_and_reclaim();
    }

    uint64_t current_epoch() const {
        return global_epoch_.load(std::memory_order_acquire);
    }

private:
    struct RetiredNode {
        void* ptr;
        void (*deleter)(void*);
        uint64_t retire_epoch;
    };

    struct alignas(64) Participant {
        std::atomic<uint64_t> local_epoch{0};
        std::atomic<uint32_t> pin_count{0};
        std::atomic<bool> active{false};
        std::vector<RetiredNode> local_retired;
    };

    class ThreadSlot {
    public:
        ThreadSlot() {
            EBRManager::get_instance().register_thread(&participant_);
        }

        ~ThreadSlot() {
            EBRManager::get_instance().unregister_thread(&participant_);
        }

        Participant& participant() { return participant_; }

    private:
        Participant participant_;
    };

    EBRManager() = default;
    ~EBRManager() {
        // 进程退出时尽力清理。
        reclaim_all();
    }

    EBRManager(const EBRManager&) = delete;
    EBRManager& operator=(const EBRManager&) = delete;

    static Participant& local_participant() {
        static thread_local ThreadSlot slot;
        return slot.participant();
    }

    void register_thread(Participant* participant) {
        std::lock_guard<std::mutex> lock(participants_mutex_);
        participants_.push_back(participant);
        participant->local_retired.reserve(kLocalBatchThreshold);
    }

    void unregister_thread(Participant* participant) {
        // 先将本地退休对象刷入全局队列，再从参与者列表移除。
        flush_local_retired(*participant);

        {
            std::lock_guard<std::mutex> lock(participants_mutex_);
            for (auto it = participants_.begin(); it != participants_.end(); ++it) {
                if (*it == participant) {
                    participants_.erase(it);
                    break;
                }
            }
        }

        try_advance_epoch_and_reclaim();
    }

    void maybe_flush_local_retired(Participant& participant) {
        if (participant.local_retired.size() >= (kLocalBatchThreshold / 2U)) {
            flush_local_retired(participant);
        }
    }

    void flush_local_retired(Participant& participant) {
        if (participant.local_retired.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(retire_mutex_);
        for (const RetiredNode& node : participant.local_retired) {
            global_retired_[bucket_index(node.retire_epoch)].push_back(node);
        }
        participant.local_retired.clear();
    }

    void try_advance_epoch_and_reclaim() {
        uint64_t observed = global_epoch_.load(std::memory_order_acquire);
        if (can_advance_epoch(observed)) {
            (void)global_epoch_.compare_exchange_strong(
                observed,
                observed + 1,
                std::memory_order_acq_rel,
                std::memory_order_acquire);
        }

        const uint64_t current = global_epoch_.load(std::memory_order_acquire);
        if (current < 2U) {
            return;
        }

        const uint64_t reclaim_epoch = current - 2U;
        reclaim_epoch_bucket(reclaim_epoch);
    }

    bool can_advance_epoch(uint64_t observed_epoch) {
        std::lock_guard<std::mutex> lock(participants_mutex_);
        for (Participant* participant : participants_) {
            if (!participant->active.load(std::memory_order_acquire)) {
                continue;
            }

            const uint64_t e = participant->local_epoch.load(std::memory_order_acquire);
            if (e != observed_epoch) {
                return false;
            }
        }
        return true;
    }

    void reclaim_epoch_bucket(uint64_t safe_epoch) {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        std::vector<RetiredNode>& bucket = global_retired_[bucket_index(safe_epoch)];

        std::size_t write = 0;
        for (std::size_t read = 0; read < bucket.size(); ++read) {
            RetiredNode& node = bucket[read];
            if (node.retire_epoch <= safe_epoch) {
                node.deleter(node.ptr);
            } else {
                if (write != read) {
                    bucket[write] = node;
                }
                ++write;
            }
        }
        bucket.resize(write);
    }

    void reclaim_all() {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        for (auto& bucket : global_retired_) {
            for (RetiredNode& node : bucket) {
                node.deleter(node.ptr);
            }
            bucket.clear();
        }
    }

    static void free_deleter(void* ptr) {
        std::free(ptr);
    }

    template <typename T>
    static void typed_delete(void* ptr) {
        delete static_cast<T*>(ptr);
    }

    static constexpr std::size_t kEpochBuckets = 3;
    static constexpr std::size_t kLocalBatchThreshold = 64;

    static std::size_t bucket_index(uint64_t epoch) {
        return static_cast<std::size_t>(epoch % kEpochBuckets);
    }

    std::atomic<uint64_t> global_epoch_{1};

    mutable std::mutex participants_mutex_;
    std::vector<Participant*> participants_;

    std::mutex retire_mutex_;
    std::array<std::vector<RetiredNode>, kEpochBuckets> global_retired_{};
};

} // namespace vector_search
