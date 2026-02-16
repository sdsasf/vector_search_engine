#include <iostream>
#include <brpc/server.h>
#include <bvar/bvar.h>
#include <gflags/gflags.h>
#include <butil/time.h>
#include "vector_search.pb.h"
#include "hnsw_index.h"
#include "utils.h"

using namespace vector_search;

// 定义 bvar 监控指标 (暴露在 vector_search 命名空间下)
// LatencyRecorder 会自动帮我们计算 QPS、平均延迟、P99、P999 等核心指标
bvar::LatencyRecorder g_search_latency("vector_search", "search_latency");
// 记录发生错误的请求数
bvar::Adder<int> g_search_error("vector_search", "search_error");

// 实现 Protobuf 中定义的 Service
class VectorSearchServiceImpl : public pb::VectorSearchService {
public:
    VectorSearchServiceImpl(HnswIndex* index) : index_(index) {}

    // 核心的 Search 接口实现
    virtual void Search(google::protobuf::RpcController* cntl_base,
                        const pb::SearchRequest* request,
                        pb::SearchResponse* response,
                        google::protobuf::Closure* done) {
        // brpc 独有的 RAII 机制，确保 done->Run() 会在函数结束时被调用
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);

        // 记录开始时间
        int64_t start_time_us = butil::gettimeofday_us();

        // 校验输入
        if (request->query_vector_size() != 128) { // 假设固定 128 维
            response->set_code(-1);
            response->set_message("Invalid query vector dimension.");
            g_search_error << 1; 
            return;
        }

        // 提取数据并调用我们底层的无锁 HNSW 引擎
        std::vector<float> query(request->query_vector().begin(), request->query_vector().end());
        
        try {
            auto results = index_->search_knn(query.data(), request->k(), request->ef_search());
            
            // 组装响应
            for (auto id : results) {
                response->add_ids(id);
            }
            response->set_code(0);
            response->set_message("Success");

        } catch (const std::exception& e) {
            response->set_code(-2);
            response->set_message(e.what());
            g_search_error << 1;
        }

        // 记录单次请求的延迟，bvar 会在后台自动聚合 QPS 和 P99 分布！
        int64_t cost_us = butil::gettimeofday_us() - start_time_us;
        g_search_latency << cost_us; 
    }

private:
    HnswIndex* index_;
};

int main(int argc, char* argv[]) {
    // 1. 初始化并加载底层 SIFT 数据作为底库 (复用之前的逻辑)
    std::cout << "Loading base data into HNSW Engine..." << std::endl;
    size_t dim, num;
    auto base_data = load_fvecs("../data/sift/sift_base.fvecs", dim, num);
    HnswIndex index(dim, num, 16, 200);
    
    for (size_t i = 0; i < num; ++i) {
        index.insert(base_data.data() + i * dim, i);
        // 如果想看进度，可以加上打印
        if (i % 200000 == 0) {
            std::cout << "Server initializing: " << i << " / " << num << " inserted." << std::endl;
        }
    }
    std::cout << "Engine ready." << std::endl;

    // 2. 启动 bRPC 服务端
    brpc::Server server;
    VectorSearchServiceImpl vector_service(&index);

    if (server.AddService(&vector_service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        std::cerr << "Fail to add service" << std::endl;
        return -1;
    }

    // 启动在 8000 端口
    brpc::ServerOptions options;
    options.idle_timeout_sec = -1;
    if (server.Start(8000, &options) != 0) {
        std::cerr << "Fail to start VectorSearchServer" << std::endl;
        return -1;
    }

    std::cout << "VectorSearchServer is successfully running on port 8000" << std::endl;
    std::cout << "You can view built-in metrics at: http://localhost:8000/brpc_metrics" << std::endl;

    // 等待直到按下 Ctrl-C
    server.RunUntilAskedToQuit();
    return 0;
}