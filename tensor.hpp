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


    // Matrix Multiplication (dot product rule)
    [[nodiscard]] static Tensor matmul(const Tensor &A, const Tensor &B) {
        if (A.cols != B.rows) throw std::invalid_argument("Dimension mismatch for matmul");
        Tensor C(A.rows, B.cols);
        for (std::size_t i = 0; i < A.rows, ++i) {
            for (std::size_t k = 0; k < A.cols; ++k) {
                double a_ik = A(i, k);  // Cache this value once
                for (std::size_t j = 0; j < B.cols; ++j) {
                    C(i, j) += a_ik * B(k, j);
                }
            }
        }
        return C;
    }

    // Transpose of the tensor
    // This is used during backpropagation
    [[nodiscard]] Tensor transpose() const {
        Tensor C(cols, rows);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; i++)
                C(j, i) = data[i * cols + j];
        return C;
    }

    // return sums of cols for bias gradients calculations
    [[nodiscard]] std::vector<double> sum_rows() const {
        std::vector<double> sum(cols, 0.0);
        for (std::size_t j = 0; j < cols; ++j)
            for (std::size_t i = 0; i < rows; ++i)
                sums[j] += data[i * cols + j];
        return sums;
    }

};