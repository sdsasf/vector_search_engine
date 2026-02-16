cmake_minimum_required(VERSION 3.14)
project(VectorSearchEngine CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启 O3 优化，并显式启用 AVX2 和 FMA 指令集支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")

# 添加头文件搜索路径
include_directories(include)

# 编译核心库
add_library(core_distance src/distance.cpp)

# 添加 benchmark 目录
add_subdirectory(benchmark)