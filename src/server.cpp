#include <iostream>
#include <brpc/server.h>
#include <bvar/bvar.h>
#include <butil/time.h>
#include <gflags/gflags.h>
#include "vector_search.pb.h"
#include "engine.h" // 替换 hnsw_index.h
#include "utils.h"

using namespace vector_search;

bvar::LatencyRecorder g_search_latency("vector_search", "search_latency");
bvar::LatencyRecorder g_insert_latency("vector_search", "insert_latency"); // 新增写入监控

class VectorSearchServiceImpl : public pb::VectorSearchService {
public:
    // 注意：这里换成了 VectorEngine
    VectorSearchServiceImpl(VectorEngine* engine) : engine_(engine) {}

    virtual void Search(google::protobuf::RpcController* cntl_base,
                        const pb::SearchRequest* request,
                        pb::SearchResponse* response,
                        google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        int64_t start_time_us = butil::gettimeofday_us();

        if (request->query_vector_size() != 128) {
            response->set_code(-1);
            return;
        }

        std::vector<float> query(request->query_vector().begin(), request->query_vector().end());
        try {
            // 调用多路归并的 engine_->search_knn
            auto results = engine_->search_knn(query.data(), request->k(), request->ef_search());
            for (auto id : results) response->add_ids(id);
            response->set_code(0);
        } catch (...) {
            response->set_code(-2);
        }
        g_search_latency << (butil::gettimeofday_us() - start_time_us); 
    }

    // 实现新增加的 Insert 接口
    virtual void Insert(google::protobuf::RpcController* cntl_base,
                        const pb::InsertRequest* request,
                        pb::InsertResponse* response,
                        google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        int64_t start_time_us = butil::gettimeofday_us();

        if (request->vector_size() != 128) {
            response->set_code(-1);
            return;
        }

        std::vector<float> vec(request->vector().begin(), request->vector().end());
        try {
            // 将写请求直接打入极速前台 Buffer
            engine_->insert(vec.data(), request->id());
            response->set_code(0);
        } catch (...) {
            response->set_code(-2);
        }
        g_insert_latency << (butil::gettimeofday_us() - start_time_us);
    }

private:
    VectorEngine* engine_;
};

int main(int argc, char* argv[]) {
    std::cout << "Loading base data into Vector Engine..." << std::endl;
    size_t dim, num;
    auto base_data = load_fvecs("../data/sift/sift_base.fvecs", dim, num);
    
    // 初始化我们的动静分离 Engine (容量百万，Buffer容量5万)
    VectorEngine engine(dim, 1000000, 16, 200, 50000, 2);
    
    // Bulk Load 模式：利用所有 CPU 核心，直接并发写入底层图
    std::cout << "Starting Bulk Load Phase (Using all CPU cores)..." << std::endl;
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> build_threads;
    std::atomic<size_t> built_count{0};
    
    int64_t start_build = butil::gettimeofday_us();

    for (int t = 0; t < num_threads; ++t) {
        build_threads.emplace_back([&, t]() {
            // 每个线程跳跃式地分担底库数据
            for (size_t i = t; i < num; i += num_threads) {
                
                // 绕过 engine.insert 的前台缓冲，调用专供 Bulk Load 的接口，直接原地建图
                engine.get_raw_index()->insert_bulk(base_data.data() + i * dim, i);
                
                size_t current = built_count.fetch_add(1, std::memory_order_relaxed);
                // 打印进度条
                if ((current + 1) % 100000 == 0) {
                    std::cout << "Actually built into graph: " << (current + 1) << " / " << num << std::endl;
                }
            }
        });
    }

    // 等待所有建图线程完成
    for (auto& t : build_threads) {
        t.join();
    }
    
    double build_time = (butil::gettimeofday_us() - start_build) / 1000000.0;
    std::cout << "Bulk Load completely finished in " << build_time << " seconds." << std::endl;
    std::cout << "Engine transition to Streaming Mode. Ready for RPC requests." << std::endl;

    brpc::Server server;
    VectorSearchServiceImpl vector_service(&engine);

    if (server.AddService(&vector_service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) return -1;

    brpc::ServerOptions options;
    options.idle_timeout_sec = -1;
    if (server.Start(8000, &options) != 0) return -1;

    std::cout << "VectorSearchServer running on port 8000" << std::endl;
    server.RunUntilAskedToQuit();
    return 0;
}