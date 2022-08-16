#include "MutationRateContainer.hpp"

namespace neat {
    void MutationRateContainer::read(std::ifstream& o){
		o >> this->connection_mutate_chance;
		o >> this->perturb_chance;
		o >> this->crossover_chance;
		o >> this->link_mutation_chance;
		o >> this->node_mutation_chance;
		o >> this->bias_mutation_chance;
		o >> this->step_size;
		o >> this->disable_mutation_chance;
		o >> this->enable_mutation_chance;
	}

	void MutationRateContainer::write(std::ofstream& o, std::string prefix){
		o << prefix << this->connection_mutate_chance << std::endl;
		o << prefix << this->perturb_chance << std::endl;
		o << prefix << this->crossover_chance << std::endl;
		o << prefix << this->link_mutation_chance << std::endl;
		o << prefix << this->node_mutation_chance << std::endl;
		o << prefix << this->bias_mutation_chance << std::endl;
		o << prefix << this->step_size << std::endl;
		o << prefix << this->disable_mutation_chance << std::endl;
		o << prefix << this->enable_mutation_chance << std::endl;
	}
}
