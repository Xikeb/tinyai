#pragma once

#include <unordered_map>

#include "MutationRateContainer.hpp"
#include "NetworkInfoContainer.hpp"
#include "Gene.hpp"

namespace neat {
	class Genotype {
	private:
		Genotype() = default;

	public:
		MutationRateContainer mutation_rates;
		NetworkInfoContainer network_info{};

		unsigned int fitness = 0;
		unsigned int adjusted_fitness = 0;
		unsigned int global_rank = 0;
		unsigned int max_neuron{};
		unsigned int can_be_recurrent = false;

        //InnovationId to Gene
		std::unordered_map<unsigned int, Gene> genes;

		Genotype(NetworkInfoContainer& info, MutationRateContainer& rates):
        mutation_rates(rates), network_info(info), max_neuron(info.functional_nodes) {
		}

		Genotype(const Genotype&) = default;

        static double disjoint(Genotype const &g1, Genotype const &g2);
        static double weights(Genotype const &g1, Genotype const &g2);
	};
}