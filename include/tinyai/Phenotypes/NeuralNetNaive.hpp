//
// Created by eliord on 7/24/22.
//

#pragma once

#include <vector>
#include <cstdlib>
#include <string>

#include "tinyai/Genotype.hpp"
#include "tinyai/Phenotypes/Neuron.hpp"

namespace ann {
    class NeuralNetNaive {
    private:
		std::vector<Neuron> nodes;
		bool recurrent = false;

		std::vector<size_t> input_nodes;
		std::vector<size_t> bias_nodes;
		std::vector<size_t> output_nodes;

        std::vector<size_t> nonrecurrentEvaluationOrder;

	public:
		void from_genome(const neat::Genotype& a);

		void evaluate(const std::vector<double>& input, std::vector<double>& output);

        constexpr static bool isRecurrent() noexcept {
            return false;
        }

		void import_fromfile(const std::string& filename);
		void export_tofile(const std::string& filename);
    };
}
