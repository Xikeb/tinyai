#pragma once

#include <fstream>

namespace neat {
    struct SpeciatingParameterContainer {
		unsigned int population = 240;
		double delta_disjoint = 2.0;
		double delta_weights = 0.4;
		double delta_threshold = 1.3;
		unsigned int stale_species = 15;

		void read(std::ifstream& o);
		void write(std::ofstream& o, std::string prefix);
	};
}