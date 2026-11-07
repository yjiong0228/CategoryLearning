#pragma once
#include <cstdint>
#include <random>
#include <span>
#include <string>
namespace gm {
std::uint64_t derive_seed(std::uint64_t root, const std::string& purpose);
class Rng {
    std::mt19937_64 engine_;
public:
    explicit Rng(std::uint64_t seed):engine_(seed){}
    double uniform();
    double normal();
    std::size_t integer(std::size_t upper);
    std::size_t categorical(std::span<const double> probabilities);
};
}
