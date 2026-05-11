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

};