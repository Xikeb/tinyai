//
// Created by eliord on 4/10/22.
//

#ifndef TINYAI_NEURALNET_HPP
#define TINYAI_NEURALNET_HPP

#include <vector>
#include <unordered_map>
#include <cmath>
#include <stack>
#include <queue>
#include <iostream>
#include <fstream>

#include "tinyai/Phenotypes/Neuron.hpp"
#include "tinyai/Genotype.hpp"

namespace ann {
    class NeuralNet {

	private:
		std::vector<Neuron> nodes;
		bool recurrent = false;

		std::vector<size_t> input_nodes;
		std::vector<size_t> bias_nodes;
		std::vector<size_t> output_nodes;

		double sigmoid(double x){
			return 2.0/(1.0 + std::exp(-4.9*x)) - 1;
		}

		void evaluate_nonrecurrent(const std::vector<double>& input, std::vector<double>& output);

		void evaluate_recurrent(const std::vector<double>& input, std::vector<double>& output);


	public:
		NeuralNet() = default;

		void from_genome(const neat::Genotype& a);

		void evaluate(const std::vector<double>& input, std::vector<double>& output);

		void import_fromfile(const std::string& filename);
		void export_tofile(const std::string& filename);
    };
}



#endif //TINYAI_NEURALNET_HPP
