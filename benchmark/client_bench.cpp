#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <unordered_set>
#include <vector>
#include <brpc/channel.h>
#include <bvar/bvar.h>
#include "vector_search.pb.h"
#include "utils.h"

using namespace vector_search;

// 独立的监控大盘：分别记录搜索和插入的端到端延迟
bvar::LatencyRecorder g_client_search_latency("vector_client", "search_latency");
bvar::LatencyRecorder g_client_insert_latency("vector_client", "insert_latency");

int main(int argc, char* argv[]) {
    std::cout << "Loading Query Data and Groundtruth for testing..." << std::endl;
    size_t query_dim, query_num;
    auto query_data = load_fvecs("../data/sift/sift_query.fvecs", query_dim, query_num);

    size_t gt_dim, gt_num;
    auto groundtruth = load_ivecs("../data/sift/sift_groundtruth.ivecs", gt_dim, gt_num);
    std::cout << "Data loaded. Initializing 12-Thread Attack (6 Search + 6 Insert)!" << std::endl;

    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_BAIDU_STD; 
    options.connection_type = brpc::CONNECTION_TYPE_POOLED; 
    options.timeout_ms = 2000; // 混合读写下，超时放宽到 2 秒
    options.max_retry = 3;

    if (channel.Init("127.0.0.1:8000", &options) != 0) {
        std::cerr << "Fail to initialize channel" << std::endl;
        return -1;
    }

    int num_search_threads = 6;
    int num_insert_threads = 6;
    std::vector<std::thread> threads;
    
    std::atomic<int> search_hits{0};
    std::atomic<int> search_success{0};
    std::atomic<int> insert_success{0};
    
    int k = 10;
    int ef_search = 50;
    uint32_t start_new_id = 1000000; // 新插入的向量 ID 从 100 万开始

    auto start_time = std::chrono::high_resolution_clock::now();

    // ==========================================
    // 启动 6 个疯狂的【搜索线程】
    // ==========================================
    for (int t = 0; t < num_search_threads; ++t) {
        threads.emplace_back([&, t]() {
            pb::VectorSearchService_Stub stub(&channel);
            // 每个线程跑满所有的 query 数据
            for (size_t i = t; i < query_num; i += num_search_threads) {
                pb::SearchRequest request;
                pb::SearchResponse response;
                brpc::Controller cntl;

                request.set_k(k);
                request.set_ef_search(ef_search);
                const float* vec_start = query_data.data() + i * query_dim;
                for (size_t j = 0; j < query_dim; ++j) request.add_query_vector(vec_start[j]);

                int64_t start_us = butil::gettimeofday_us();
                stub.Search(&cntl, &request, &response, NULL);
                int64_t cost_us = butil::gettimeofday_us() - start_us;

                if (!cntl.Failed() && response.code() == 0) {
                    search_success.fetch_add(1, std::memory_order_relaxed);
                    g_client_search_latency << cost_us;

                    int hits = 0;
                    std::unordered_set<uint32_t> gt_set(groundtruth[i].begin(), groundtruth[i].begin() + k);
                    for (int j = 0; j < response.ids_size(); ++j) {
                        if (gt_set.count(response.ids(j))) hits++;
                    }
                    search_hits.fetch_add(hits, std::memory_order_relaxed);
                }
            }
        });
    }

    // ==========================================
    // 启动 6 个疯狂的【写入线程】
    // ==========================================
    for (int t = 0; t < num_insert_threads; ++t) {
        threads.emplace_back([&, t]() {
            pb::VectorSearchService_Stub stub(&channel);
            int total_inserts = 50000 / num_insert_threads; 
            
            // 为当前写线程准备一个独立的随机数发生器
            unsigned int seed = 10086 + t; 

            for (int i = 0; i < total_inserts; ++i) {
                pb::InsertRequest request;
                pb::InsertResponse response;
                brpc::Controller cntl;

                // 【修复核心】：生成完全随机的噪声向量，绝不污染真实的查询空间
                for (size_t j = 0; j < query_dim; ++j) {
                    // 生成 1000.0 到 2000.0 之间的巨大随机数
                    // SIFT 原始数据的值通常在 0~255 左右，这样新数据会被远远推开
                    float noise_val = 1000.0f + ((float)rand_r(&seed) / RAND_MAX) * 1000.0f;
                    request.add_vector(noise_val);
                }
                
                request.set_id(start_new_id + t * total_inserts + i);

                int64_t start_us = butil::gettimeofday_us();
                stub.Insert(&cntl, &request, &response, NULL);
                int64_t cost_us = butil::gettimeofday_us() - start_us;

                if (!cntl.Failed() && response.code() == 0) {
                    insert_success.fetch_add(1, std::memory_order_relaxed);
                    g_client_insert_latency << cost_us;
                }
            }
        });
    }

    // 等待所有线程完成厮杀
    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    double search_qps = search_success.load() / total_time;
    double insert_qps = insert_success.load() / total_time;
    double recall = (double)search_hits.load() / (search_success.load() * k);

    std::cout << "\n=============================================" << std::endl;
    std::cout << "Mixed Workload Benchmark Results (6R/6W)" << std::endl;
    std::cout << "Total Time        : " << total_time << " seconds" << std::endl;
    std::cout << "Search QPS        : " << search_qps << " req/s" << std::endl;
    std::cout << "Insert QPS        : " << insert_qps << " req/s" << std::endl;
    std::cout << "Combined QPS      : " << search_qps + insert_qps << " req/s" << std::endl;
    std::cout << "Recall@" << k << "         : " << recall * 100.0 << " %" << std::endl;
    std::cout << "=============================================\n" << std::endl;
    
    // 留给 bvar 后台线程 1.5 秒的时间去汇总 P99 极值
    std::cout << "Waiting for bvar background thread to aggregate percentiles..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1500)); 

    std::cout << "\n[Search] Latency Stats:" << std::endl;
    std::cout << "Average Latency   : " << g_client_search_latency.latency() << " us" << std::endl;
    std::cout << "P99 Latency       : " << g_client_search_latency.latency_percentiles()[2] << " us" << std::endl;
    std::cout << "P999 Latency      : " << g_client_search_latency.latency_percentiles()[3] << " us" << std::endl;

    std::cout << "\n[Insert] Latency Stats:" << std::endl;
    std::cout << "Average Latency   : " << g_client_insert_latency.latency() << " us" << std::endl;
    std::cout << "P99 Latency       : " << g_client_insert_latency.latency_percentiles()[2] << " us" << std::endl;
    std::cout << "P999 Latency      : " << g_client_insert_latency.latency_percentiles()[3] << " us" << std::endl;

    return 0;
}