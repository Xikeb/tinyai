#include "InnovationContainer.hpp"

namespace neat {
    unsigned int InnovationContainer::add_gene(Gene& g) {
            auto [iter, inserted] = track.insert({std::make_pair(g.from_node, g.to_node), _number+1});

            if (inserted) {
                ++_number;
            }

            return iter->second;

			auto it = track.find(std::make_pair(g.from_node, g.to_node));
			if (it == track.end())
				return track[std::make_pair(g.from_node, g.to_node)] = ++_number;
			else
				return (*it).second;
		}

}

