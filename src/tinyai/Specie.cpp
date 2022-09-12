#include "tinyai/Specie.hpp"

namespace neat {
    void Specie::calculate_average_fitness() {
		unsigned int total = 0;
		for (auto & genome : this->genomes)
			total += genome.global_rank;
		this->average_fitness = total / this->genomes.size();
	}
}
