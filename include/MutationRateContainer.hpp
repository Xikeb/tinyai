#pragma once

#include <fstream>

namespace neat {
    struct MutationRateContainer {
		double connection_mutate_chance = 0.25;
		double perturb_chance = 0.90;
		double crossover_chance = 0.75;
		// double link_mutation_chance = 2.0;
		double link_mutation_chance = 2.5;
		double node_mutation_chance = 0.50;
		double bias_mutation_chance = 0.40;
		double step_size = 0.1;
		double disable_mutation_chance = 0.4;
		double enable_mutation_chance = 0.2;

		void read(std::ifstream& o);
		void write(std::ofstream& o, std::string prefix);
	};
};