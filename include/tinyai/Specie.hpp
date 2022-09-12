#pragma once

#include <vector>

#include "tinyai/Genotype.hpp"

namespace neat {
    /* a Specie is group of genomes which differences is smaller than some threshold */
	struct Specie {
		unsigned int top_fitness = 0;
		unsigned int average_fitness = 0;
		unsigned int staleness = 0;

	#ifdef GIVING_NAMES_FOR_SPECIES
		std::string name;
	#endif
		std::vector<Genotype> genomes;

        Specie() = default;
        explicit Specie(Genotype const &progenitor): genomes{progenitor} {}
        explicit Specie(Genotype &&progenitor): genomes{progenitor} {}

        void calculate_average_fitness();
	};
}