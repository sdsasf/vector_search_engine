#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

namespace vector_search {

// 读取 .fvecs 格式的文件 (特征向量)
inline std::vector<float> load_fvecs(const std::string& filename, size_t& dim, size_t& num) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // 读取第一个向量的维度
    int32_t d;
    input.read((char*)&d, sizeof(int32_t));
    dim = d;

    // 计算文件总大小以推断向量数量
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num = file_size / (sizeof(int32_t) + dim * sizeof(float));

    // 回到开头，开始读取
    input.seekg(0, std::ios::beg);
    std::vector<float> data(num * dim);

    for (size_t i = 0; i < num; ++i) {
        input.read((char*)&d, sizeof(int32_t));
        if (d != dim) {
            throw std::runtime_error("Dimension mismatch in file!");
        }
        input.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
    return data;
}

// 读取 .ivecs 格式的文件 (GroundTruth 答案)
inline std::vector<std::vector<uint32_t>> load_ivecs(const std::string& filename, size_t& dim, size_t& num) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int32_t d;
    input.read((char*)&d, sizeof(int32_t));
    dim = d;

    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num = file_size / (sizeof(int32_t) + dim * sizeof(int32_t));

    input.seekg(0, std::ios::beg);
    std::vector<std::vector<uint32_t>> data(num, std::vector<uint32_t>(dim));

    for (size_t i = 0; i < num; ++i) {
        input.read((char*)&d, sizeof(int32_t));
        if (d != dim) {
            throw std::runtime_error("Dimension mismatch in file!");
        }
        input.read((char*)data[i].data(), dim * sizeof(uint32_t));
    }
    return data;
}

} // namespace vector_search