#include "SpeciatingParameterContainer.hpp"

namespace neat {
	void SpeciatingParameterContainer::read(std::ifstream& o){
		o >> this->population;
		o >> this->delta_disjoint;
		o >> this->delta_weights;
		o >> this->delta_threshold;
		o >> this->stale_species;
	}

	void SpeciatingParameterContainer::write(std::ofstream& o, std::string prefix){
		o << prefix << this->population << std::endl;
		o << prefix << this->delta_disjoint << std::endl;
		o << prefix << this->delta_weights << std::endl;
		o << prefix << this->delta_threshold << std::endl;
		o << prefix << this->stale_species << std::endl;
	}
}
