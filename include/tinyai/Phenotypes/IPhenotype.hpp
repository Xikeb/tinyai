//
// Created by eliord on 7/23/22.
//

#pragma once

#include <cmath>
#include <vector>
#include <string>

#include "tinyai/Genotype.hpp"

namespace ann {
    static double sigmoid(double x) noexcept {
        //We might not even need the std version that checks over/underflow
        return 2.0/(1.0 + std::exp(-4.9*x)) - 1;
    }

    class IPhenotype {
        void evaluate(const std::vector<double>& input, std::vector<double>& output);

        bool isRecurrent() const noexcept;
    };
}
