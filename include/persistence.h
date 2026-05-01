#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_search {

inline uint32_t persistence_checksum(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

struct WalRecordHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t type;
    uint32_t id;
    uint32_t dim;
    uint32_t payload_bytes;
    uint64_t lsn;
    uint32_t checksum;
};

class WalWriter {
public:
    static constexpr uint32_t kMagic = 0x314C4157u; // WAL1
    static constexpr uint32_t kVersion = 1;
    static constexpr uint32_t kInsert = 1;

    explicit WalWriter(const std::string& path, uint64_t start_lsn = 0)
        : path_(path), next_lsn_(start_lsn + 1) {
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path());
        out_.open(path_, std::ios::binary | std::ios::app);
        if (!out_) {
            throw std::runtime_error("Cannot open WAL: " + path_);
        }
    }

    uint64_t append_insert(uint32_t id, const float* vector, uint32_t dim) {
        const uint64_t lsn = next_lsn_++;
        WalRecordHeader header{};
        header.magic = kMagic;
        header.version = kVersion;
        header.type = kInsert;
        header.id = id;
        header.dim = dim;
        header.payload_bytes = dim * sizeof(float);
        header.lsn = lsn;
        header.checksum = persistence_checksum(vector, header.payload_bytes);

        out_.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out_.write(reinterpret_cast<const char*>(vector), header.payload_bytes);
        out_.flush();
        if (!out_) {
            throw std::runtime_error("Failed to append WAL: " + path_);
        }
        return lsn;
    }

    uint64_t current_lsn() const {
        return next_lsn_ == 0 ? 0 : next_lsn_ - 1;
    }

    void reset(uint64_t next_lsn) {
        out_.close();
        out_.open(path_, std::ios::binary | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("Cannot truncate WAL: " + path_);
        }
        next_lsn_ = next_lsn;
    }

private:
    std::string path_;
    std::ofstream out_;
    uint64_t next_lsn_;
};

template <typename Fn>
uint64_t replay_wal(const std::string& path, uint64_t after_lsn, uint32_t expected_dim, Fn&& apply) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return after_lsn;
    }

    uint64_t last_lsn = after_lsn;
    while (true) {
        WalRecordHeader header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (in.eof()) {
            break;
        }
        if (!in) {
            throw std::runtime_error("Corrupt WAL header: " + path);
        }
        if (header.magic != WalWriter::kMagic || header.version != WalWriter::kVersion ||
            header.type != WalWriter::kInsert || header.dim != expected_dim ||
            header.payload_bytes != expected_dim * sizeof(float)) {
            throw std::runtime_error("Unsupported or corrupt WAL record: " + path);
        }

        std::vector<float> vector(header.dim);
        in.read(reinterpret_cast<char*>(vector.data()), header.payload_bytes);
        if (!in) {
            throw std::runtime_error("Truncated WAL payload: " + path);
        }
        if (persistence_checksum(vector.data(), header.payload_bytes) != header.checksum) {
            throw std::runtime_error("WAL checksum mismatch: " + path);
        }

        if (header.lsn > after_lsn) {
            apply(header.id, vector.data());
            last_lsn = header.lsn;
        }
    }
    return last_lsn;
}

} // namespace vector_search
