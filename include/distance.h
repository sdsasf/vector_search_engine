#pragma once
#include <cstddef>

namespace vector_search {

// 普通标量版本的 L2 距离计算
float l2_distance_scalar(const float* a, const float* b, size_t dim);

// 基于 AVX2 和 FMA 指令集优化的 L2 距离计算
float l2_distance_avx2(const float* a, const float* b, size_t dim);

} // namespace vector_search