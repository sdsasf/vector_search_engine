#include "distance.h"
#include <immintrin.h> // Intel AVX 指令集头文件

namespace vector_search {

float l2_distance_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float l2_distance_avx2(const float* a, const float* b, size_t dim) {
    __m256 sum_vec = _mm256_setzero_ps(); // 初始化一个全 0 的 256 位寄存器 (存 8 个 float)
    
    size_t i = 0;
    // 每次处理 8 个 float (256 bits)
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i); // 不要求内存对齐的加载
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb); // 向量减法
        // FMA (Fused Multiply-Add): sum_vec = diff * diff + sum_vec
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec); 
    }

    // 将 8 个部分和汇总到一个 float 数组中
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum_vec);
    float res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    // 处理剩余无法凑齐 8 个的尾部元素
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }

    return res;
}

} // namespace vector_search