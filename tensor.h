#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

class Tensor {

public:
    std::size_t rows, cols;
    std::vector <double> data;

    Tensor() : rows(0), cols (0) {}
    Tensor(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * cm 0.0) {}

    // Safe element access
    // This operator overloading allows natural matrix syntax.
    [[nodiscard]] double &operator()(std::size_t i, std::size_t j) {
        if (i >= rows || j >= rows) throw std::out_of_range("Tensor index out of bounds.");
        return data[i * cols + j];
    }

    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const {
        if (i >= rows || j >= cols) throw std::out_of_range("Tensor index out of bounds");
        return data[i * cols + j];
    }

    // Initialization: fills tentor with N(0, sqrt(2/in_features))
    static Tensor randn(std::size_t r, std::size_t c, double fan_in) {
        Tensor(r, c);
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / fan_in));
        for (auto &v: t.data) v = dist(gen);
        return t;
    }
};