#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <unordered_set>
#include "hnsw_index.h"
#include "utils.h"

using namespace vector_search;

int main() {
    std::cout << "Loading SIFT1M Dataset..." << std::endl;
    
    size_t base_dim, base_num;
    auto base_data = load_fvecs("../data/sift/sift_base.fvecs", base_dim, base_num);
    std::cout << "Base data loaded: " << base_num << " vectors, dim=" << base_dim << std::endl;

    size_t query_dim, query_num;
    auto query_data = load_fvecs("../data/sift/sift_query.fvecs", query_dim, query_num);
    std::cout << "Query data loaded: " << query_num << " vectors, dim=" << query_dim << std::endl;

    size_t gt_dim, gt_num;
    auto groundtruth = load_ivecs("../data/sift/sift_groundtruth.ivecs", gt_dim, gt_num);
    std::cout << "Groundtruth loaded." << std::endl;

    // 初始化 HNSW 索引
    // 参数：M=16 (每层最大连接数), ef_construction=200 (建图搜索深度，越大图质量越高)
    HnswIndex index(base_dim, base_num, 16, 200);

    // --------------------------------------------------------
    // 测试 1：多线程并发无锁建图
    // --------------------------------------------------------
    std::cout << "\nStarting multi-threaded lock-free insertion..." << std::endl;
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::atomic<size_t> insert_count{0};

    auto start_build = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = t; i < base_num; i += num_threads) {
                index.insert(base_data.data() + i * base_dim, i);
                insert_count.fetch_add(1, std::memory_order_relaxed);
                
                // 打印进度
                if (i % 50000 == 0 && t == 0) {
                    std::cout << "Inserted " << insert_count.load() << " / " << base_num << " vectors..." << std::endl;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end_build - start_build).count();
    std::cout << "Build time: " << build_time << " seconds. (Throughput: " 
              << base_num / build_time << " vectors/sec)" << std::endl;

    // --------------------------------------------------------
    // 测试 2：并发查询与召回率 (Recall@10) 计算
    // --------------------------------------------------------
    std::cout << "\nStarting search benchmark..." << std::endl;
    int k = 10;
    int ef_search = 100; // 探索深度，越大越准，但越慢
    std::atomic<int> total_hits{0};
    
    auto start_search = std::chrono::high_resolution_clock::now();

    threads.clear();
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = t; i < query_num; i += num_threads) {
                // 执行查询
                auto results = index.search_knn(query_data.data() + i * query_dim, k, ef_search);
                
                // 计算与 groundtruth 的交集
                int hits = 0;
                std::unordered_set<uint32_t> gt_set(groundtruth[i].begin(), groundtruth[i].begin() + k);
                for (auto res_id : results) {
                    if (gt_set.count(res_id)) {
                        hits++;
                    }
                }
                total_hits.fetch_add(hits, std::memory_order_relaxed);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_search = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double>(end_search - start_search).count();
    
    double qps = query_num / search_time;
    double recall = (double)total_hits / (query_num * k);

    std::cout << "=============================" << std::endl;
    std::cout << "Search Parameters : k=" << k << ", ef_search=" << ef_search << std::endl;
    std::cout << "Total Search Time : " << search_time << " seconds" << std::endl;
    std::cout << "QPS (Queries/sec) : " << qps << std::endl;
    std::cout << "Recall@" << k << "         : " << recall * 100.0 << " %" << std::endl;
    std::cout << "=============================" << std::endl;

    return 0;
}