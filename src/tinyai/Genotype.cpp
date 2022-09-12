#include <cmath>

#include "tinyai/Genotype.hpp"

namespace neat {

	double Genotype::disjoint(const Genotype& g1, const Genotype& g2) {
		auto it1 = g1.genes.begin();
		auto it2 = g2.genes.begin();

		unsigned int disjoint_count = 0;
		for (; it1 != g1.genes.end(); it1++)
			if (g2.genes.find((*it1).second.innovation_num) == g2.genes.end())
				disjoint_count++;

		for (; it2 != g2.genes.end(); it2++)
			if (g1.genes.find((*it2).second.innovation_num) == g1.genes.end())
				disjoint_count++;

		return (1. * disjoint_count) / (1. * std::max(g1.genes.size(), g2.genes.size()));
	}

	double Genotype::weights(const Genotype& g1, const Genotype& g2) {
		auto it1 = g1.genes.begin();

		double sum = 0.0;
		unsigned int coincident = 0;

		for (; it1 != g1.genes.end(); it1++){
			auto it2 = g2.genes.find((*it1).second.innovation_num);
			if (it2 != g2.genes.end()){
				coincident++;
				sum += std::abs((*it1).second.weight - (*it2).second.weight);
			}
		}

		return 1. * sum / (1. * coincident);
	}
}


