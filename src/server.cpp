#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

#include <brpc/server.h>
#include <butil/time.h>
#include <bvar/bvar.h>
#include <gflags/gflags.h>

#include "engine.h"
#include "utils.h"
#include "vector_search.pb.h"

DEFINE_string(data_path, "../data/sift/sift_base.fvecs", "Path to initial .fvecs base data");
DEFINE_string(persist_dir, "../data/persistence", "Directory for WAL and snapshots");
DEFINE_uint32(dim, 128, "Vector dimension");
DEFINE_uint32(max_elements, 1000000, "Maximum number of vectors");
DEFINE_uint32(M, 16, "HNSW max neighbors per upper layer");
DEFINE_uint32(ef_construction, 200, "HNSW construction ef");
DEFINE_uint32(buffer_capacity, 50000, "Streaming write buffer capacity");
DEFINE_uint32(bg_threads, 2, "Background flush threads");

using namespace vector_search;

bvar::LatencyRecorder g_search_latency("vector_search", "search_latency");
bvar::LatencyRecorder g_insert_latency("vector_search", "insert_latency");

class VectorSearchServiceImpl : public pb::VectorSearchService {
public:
    explicit VectorSearchServiceImpl(VectorEngine* engine) : engine_(engine) {}

    void Search(google::protobuf::RpcController* cntl_base,
                const pb::SearchRequest* request,
                pb::SearchResponse* response,
                google::protobuf::Closure* done) override {
        (void)cntl_base;
        brpc::ClosureGuard done_guard(done);
        int64_t start_time_us = butil::gettimeofday_us();

        if (request->query_vector_size() != static_cast<int>(FLAGS_dim)) {
            response->set_code(-1);
            response->set_message("dimension mismatch");
            return;
        }

        std::vector<float> query(request->query_vector().begin(), request->query_vector().end());
        try {
            auto results = engine_->search_knn(query.data(), request->k(), request->ef_search());
            for (auto id : results) response->add_ids(id);
            response->set_code(0);
        } catch (const std::exception& e) {
            response->set_code(-2);
            response->set_message(e.what());
        } catch (...) {
            response->set_code(-2);
            response->set_message("unknown search error");
        }
        g_search_latency << (butil::gettimeofday_us() - start_time_us);
    }

    void Insert(google::protobuf::RpcController* cntl_base,
                const pb::InsertRequest* request,
                pb::InsertResponse* response,
                google::protobuf::Closure* done) override {
        (void)cntl_base;
        brpc::ClosureGuard done_guard(done);
        int64_t start_time_us = butil::gettimeofday_us();

        if (request->vector_size() != static_cast<int>(FLAGS_dim)) {
            response->set_code(-1);
            response->set_message("dimension mismatch");
            return;
        }

        std::vector<float> vec(request->vector().begin(), request->vector().end());
        try {
            engine_->insert(vec.data(), request->id());
            response->set_code(0);
        } catch (const std::exception& e) {
            response->set_code(-2);
            response->set_message(e.what());
        } catch (...) {
            response->set_code(-2);
            response->set_message("unknown insert error");
        }
        g_insert_latency << (butil::gettimeofday_us() - start_time_us);
    }

private:
    VectorEngine* engine_;
};

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    VectorEngine engine(FLAGS_dim, FLAGS_max_elements, FLAGS_M, FLAGS_ef_construction,
                        FLAGS_buffer_capacity, FLAGS_bg_threads, FLAGS_persist_dir);

    if (!engine.has_persisted_data()) {
        std::cout << "No snapshot found. Loading base data from " << FLAGS_data_path << std::endl;
        size_t dim = 0;
        size_t num = 0;
        auto base_data = load_fvecs(FLAGS_data_path, dim, num);
        if (dim != FLAGS_dim) {
            std::cerr << "Base data dimension " << dim << " does not match --dim=" << FLAGS_dim << std::endl;
            return -1;
        }
        if (num > FLAGS_max_elements) {
            std::cerr << "Base data size exceeds --max_elements" << std::endl;
            return -1;
        }

        std::cout << "Starting bulk load with " << num << " vectors" << std::endl;
        int num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> build_threads;
        std::atomic<size_t> built_count{0};
        int64_t start_build = butil::gettimeofday_us();

        for (int t = 0; t < num_threads; ++t) {
            build_threads.emplace_back([&, t]() {
                for (size_t i = t; i < num; i += num_threads) {
                    engine.get_raw_index()->insert_bulk(base_data.data() + i * dim, static_cast<uint32_t>(i));
                    size_t current = built_count.fetch_add(1, std::memory_order_relaxed);
                    if ((current + 1) % 100000 == 0) {
                        std::cout << "Built into graph: " << (current + 1) << " / " << num << std::endl;
                    }
                }
            });
        }
        for (auto& t : build_threads) t.join();

        double build_time = (butil::gettimeofday_us() - start_build) / 1000000.0;
        std::cout << "Bulk load finished in " << build_time << " seconds. Saving initial snapshot." << std::endl;
        if (engine.persistence_enabled() && !engine.save_snapshot_and_reset_wal()) {
            std::cerr << "Failed to save initial snapshot" << std::endl;
            return -1;
        }
    } else {
        std::cout << "Loaded persisted snapshot and replayed WAL." << std::endl;
    }

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
