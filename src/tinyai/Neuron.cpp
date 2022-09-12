//
// Created by eliord on 4/10/22.
//

#include "tinyai/Phenotypes/Neuron.hpp"

namespace ann {
    Neuron::~Neuron() {
        in_nodes.clear();
    }
}
