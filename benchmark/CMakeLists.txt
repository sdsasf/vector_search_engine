find_package(benchmark REQUIRED)

# 原来的距离单测
add_executable(distance_bench distance_bench.cpp)
target_link_libraries(distance_bench core_distance benchmark::benchmark pthread)

# 【新增】召回率压测
add_executable(recall_bench recall_bench.cpp)
# 链接核心库和多线程库
target_link_libraries(recall_bench core_distance pthread)