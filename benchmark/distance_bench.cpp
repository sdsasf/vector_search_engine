#include <benchmark/benchmark.h>
#include "distance.h"
#include <vector>
#include <random>

// 辅助函数：生成随机向量
std::vector<float> generate_random_vector(size_t dim) {
    std::vector<float> vec(dim);
    std::mt19937 gen(42); // 固定种子以保证测试可重复
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

// 标量版本测试
static void BM_L2DistanceScalar(benchmark::State& state) {
    size_t dim = state.range(0);
    auto vec_a = generate_random_vector(dim);
    auto vec_b = generate_random_vector(dim);

    for (auto _ : state) {
        float res = vector_search::l2_distance_scalar(vec_a.data(), vec_b.data(), dim);
        benchmark::DoNotOptimize(res); // 防止编译器把计算优化掉
    }
}

// AVX2 版本测试
static void BM_L2DistanceAVX2(benchmark::State& state) {
    size_t dim = state.range(0);
    auto vec_a = generate_random_vector(dim);
    auto vec_b = generate_random_vector(dim);

    for (auto _ : state) {
        float res = vector_search::l2_distance_avx2(vec_a.data(), vec_b.data(), dim);
        benchmark::DoNotOptimize(res);
    }
}

// 注册 Benchmark，测试 LLM 常见的向量维度：128, 512, 1024, 4096
BENCHMARK(BM_L2DistanceScalar)->Arg(128)->Arg(512)->Arg(1024)->Arg(4096);
BENCHMARK(BM_L2DistanceAVX2)->Arg(128)->Arg(512)->Arg(1024)->Arg(4096);

BENCHMARK_MAIN();