#pragma once

#include <cstdint>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>

namespace cbls {

class RNG {
public:
    explicit RNG(uint64_t seed = 42) : gen_(seed), seed_(seed) {}

    double uniform(double lo, double hi) {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(gen_);
    }

    int64_t integers(int64_t lo, int64_t hi) {
        // [lo, hi) exclusive of hi, like numpy
        std::uniform_int_distribution<int64_t> dist(lo, hi - 1);
        return dist(gen_);
    }

    double normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(gen_);
    }

    double random() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen_);
    }

    template <typename T>
    void shuffle(std::vector<T>& vec) {
        std::shuffle(vec.begin(), vec.end(), gen_);
    }

    std::vector<int32_t> permutation(int32_t n) {
        std::vector<int32_t> v(n);
        std::iota(v.begin(), v.end(), 0);
        shuffle(v);
        return v;
    }

    // Choose k elements from [0, n) without replacement
    std::vector<int32_t> choice(int32_t n, int32_t k) {
        std::vector<int32_t> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), gen_);
        pool.resize(k);
        return pool;
    }

    uint64_t seed() const { return seed_; }

    std::mt19937_64& engine() { return gen_; }

private:
    std::mt19937_64 gen_;
    uint64_t seed_;
};

}  // namespace cbls
