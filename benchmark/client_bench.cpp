#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <unordered_set>
#include <brpc/channel.h>
#include <bvar/bvar.h>
#include "vector_search.pb.h"
#include "utils.h"

using namespace vector_search;

// 客户端也使用 bvar 来记录端到端延迟
bvar::LatencyRecorder g_client_latency("vector_client", "end_to_end_latency");

int main(int argc, char* argv[]) {
    std::cout << "Loading Query Data and Groundtruth..." << std::endl;
    size_t query_dim, query_num;
    auto query_data = load_fvecs("../data/sift/sift_query.fvecs", query_dim, query_num);

    size_t gt_dim, gt_num;
    auto groundtruth = load_ivecs("../data/sift/sift_groundtruth.ivecs", gt_dim, gt_num);
    std::cout << "Data loaded. Ready to attack server!" << std::endl;

    // 1. 初始化 bRPC Channel
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_BAIDU_STD; // 使用百度标准 RPC 协议
    options.connection_type = brpc::CONNECTION_TYPE_POOLED; // 连接池模式，适合高并发
    options.timeout_ms = 1000; // 客户端超时时间设为 1000 毫秒
    options.max_retry = 3;

    if (channel.Init("127.0.0.1:8000", &options) != 0) {
        std::cerr << "Fail to initialize channel" << std::endl;
        return -1;
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::atomic<int> total_hits{0};
    std::atomic<int> success_requests{0};
    int k = 10;
    int ef_search = 50;

    std::cout << "\nStarting " << num_threads << " threads to benchmark server..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 2. 多线程并发打流
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // 每个线程创建一个 Stub (存根)
            pb::VectorSearchService_Stub stub(&channel);

            for (size_t i = t; i < query_num; i += num_threads) {
                pb::SearchRequest request;
                pb::SearchResponse response;
                brpc::Controller cntl;

                request.set_k(k);
                request.set_ef_search(ef_search);
                
                // 将 float 数组塞入 protobuf
                const float* vec_start = query_data.data() + i * query_dim;
                for (size_t j = 0; j < query_dim; ++j) {
                    request.add_query_vector(vec_start[j]);
                }

                // 发起同步 RPC 调用
                int64_t start_us = butil::gettimeofday_us();
                stub.Search(&cntl, &request, &response, NULL);
                int64_t cost_us = butil::gettimeofday_us() - start_us;

                if (!cntl.Failed() && response.code() == 0) {
                    success_requests.fetch_add(1, std::memory_order_relaxed);
                    g_client_latency << cost_us; // 记录包含网络开销的端到端延迟

                    // 计算召回率
                    int hits = 0;
                    std::unordered_set<uint32_t> gt_set(groundtruth[i].begin(), groundtruth[i].begin() + k);
                    for (int j = 0; j < response.ids_size(); ++j) {
                        if (gt_set.count(response.ids(j))) {
                            hits++;
                        }
                    }
                    total_hits.fetch_add(hits, std::memory_order_relaxed);
                } else {
                    std::cerr << "RPC failed: " << cntl.ErrorText() << std::endl;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    double qps = success_requests.load() / total_time;
    double recall = (double)total_hits.load() / (success_requests.load() * k);

    std::cout << "=============================" << std::endl;
    std::cout << "Network Benchmark Results" << std::endl;
    std::cout << "Total Requests    : " << success_requests.load() << std::endl;
    std::cout << "Total Time        : " << total_time << " seconds" << std::endl;
    std::cout << "End-to-End QPS    : " << qps << std::endl;
    std::cout << "Recall@" << k << "         : " << recall * 100.0 << " %" << std::endl;
    std::cout << "=============================" << std::endl;
    
    std::cout << "Waiting for bvar background thread to aggregate percentiles..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    // 打印 bvar 统计的 P99 和平均延迟信息
    std::cout << "\nbvar Latency Stats:" << std::endl;
    std::cout << "Average Latency   : " << g_client_latency.latency() << " us" << std::endl;
    std::cout << "P99 Latency       : " << g_client_latency.latency_percentiles()[2] << " us" << std::endl;
    std::cout << "P999 Latency      : " << g_client_latency.latency_percentiles()[3] << " us" << std::endl;
    std::cout << "Max Latency       : " << g_client_latency.max_latency() << " us" << std::endl;

    return 0;
}