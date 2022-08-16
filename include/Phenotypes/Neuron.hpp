//
// Created by eliord on 4/10/22.
//

#ifndef TINYAI_NEURON_HPP
#define TINYAI_NEURON_HPP

#include <vector>

namespace ann {
	enum type {
		RECURRENT,
		NON_RECURRENT
	};

	class Neuron {
	public:
		int type = 0; // 0 = ordinal, 1 = input, 2 = output, 3 = bias
		double value = 0.0;
		bool visited = false;
		std::vector<std::pair<size_t, double>> in_nodes;
		~Neuron();
	};
}


#endif //TINYAI_NEURON_HPP
