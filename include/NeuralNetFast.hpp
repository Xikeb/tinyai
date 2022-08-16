//
// Created by eliord on 4/10/22.
//

#ifndef TINYAI_NEURALNET_FAST_HPP
#define TINYAI_NEURALNET_FAST_HPP

#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <stack>
#include <queue>
#include <iostream>
#include <fstream>
#include <utility>
#include <chrono>


#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#include "detail.hpp"
#include "Phenotypes/Neuron.hpp"
#include "Genotype.hpp"
#include "Phenotypes/InitializationKit.hpp"

namespace ann {
    class NeuralNetFast {
    public:
        //NodeId => connectionWeight
        using Synapse = std::pair<std::size_t, double>;
	private:
        std::vector<double> nodeActivation;
        std::unordered_multimap<size_t, Synapse> links;
		bool recurrent = false;

        size_t input_offset, input_count;
        size_t bias_offset, bias_count;
        size_t output_offset, output_count;

		static double sigmoid(double x) noexcept {
			return 2.0/(1.0 + std::exp(-4.9*x)) - 1;
		}

        std::vector<size_t> evaluationOrder;
		void evaluate_nonrecurrent(const std::vector<double>& input, std::vector<double>& output);

		void evaluate_recurrent(const std::vector<double>& input, std::vector<double>& output);


	public:
		NeuralNetFast() = default;

		void from_genome(const neat::Genotype& a);
		void from_genome(const neat::Genotype& a, InitializationKit &initKit);

		void evaluate(const std::vector<double>& input, std::vector<double>& output);
    };
}



#endif //TINYAI_NEURALNET_FAST_HPP
