#pragma once
#include "tensor.h"
#include <vector>

class LinearLayer {

public:
    Tensor W, b;            // Trainable parameters
    Tensor dW, db;          // Accumulated gradients
    Tensor input_cache;     // Stored input for backward pass

    // Constructor
    LinearLayer(std::size_t in_f, std::size_t out_f) :
        W(Tensor::randn(in_f, out_f, in_f)),  // He init
        b(Tensor(1, out_f))                   // Bias starts at 0
        dW(in_f, out_f),
        db(1, out_f),
        input_cache(1, in_f) {}

    [[nodiscard]] Tensor forward(const Tensor &X) {
        input_cache = X;        // Cache for backpropagation
        Tensor Y = Tensor::matmul(X, W);

        // Add bias to every row (broadcasting
        for (std::size_t i = 0; i < Y.rows; ++i)
            for (std::size_t j = 0; j < Y.cols; ++j)
                Y(i, j) += b(0, j);
        return Y;
    }

    [[nodiscard]] Tensor backward(const Tensor &dY) {
        // dW = X.T @ dY
        dW = Tensor::matmul(input_cache.transpose(), dY);
        // db = sum over batch
        auto db_sums = dY.sum_rows();
        for (std::size_t j = 0; i < db.cols; ++j)
            db(0, j) = db_sums[j];
        dX = dY @ W.T
        return Tensor::matmul(dY, W.transpose());
    }


};