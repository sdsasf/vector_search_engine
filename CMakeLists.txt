cmake_minimum_required(VERSION 3.14)
project(VectorSearchEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")

include_directories(include)

# Ñ°ÕÒ Protobuf ºÍ bRPC
find_package(Protobuf REQUIRED)
find_path(BRPC_INCLUDE_PATH NAMES brpc/server.h)
find_library(BRPC_LIB NAMES brpc)
include_directories(${BRPC_INCLUDE_PATH})

# ±àÒë Protobuf ÎÄ¼þ
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/proto)
execute_process(
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/proto --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto ${CMAKE_CURRENT_SOURCE_DIR}/proto/vector_search.proto
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/proto)

# ±àÒëµ×²ãºËÐÄ¿â
add_library(core_distance src/distance.cpp)

# ±àÒë bRPC ·þÎñ¶Ë
add_executable(vector_server src/server.cpp ${CMAKE_CURRENT_BINARY_DIR}/proto/vector_search.pb.cc)
target_link_libraries(vector_server core_distance ${BRPC_LIB} protobuf gflags leveldb ssl crypto pthread)

add_subdirectory(benchmark)

# ±àÒë bRPC Ñ¹²â¿Í»§¶Ë
add_executable(client_bench benchmark/client_bench.cpp ${CMAKE_CURRENT_BINARY_DIR}/proto/vector_search.pb.cc)
target_link_libraries(client_bench ${BRPC_LIB} protobuf gflags leveldb ssl crypto pthread)