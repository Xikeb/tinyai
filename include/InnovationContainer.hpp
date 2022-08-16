#pragma once

#include <unordered_map>

#include "Gene.hpp"

namespace std {
    template<>
    struct hash<std::pair<unsigned int, unsigned int>> {
        [[gnu::const]]
        /*constexpr */bool operator()(std::pair<unsigned int, unsigned int> const &p) const noexcept {
            constexpr std::hash<unsigned int> subhasher{};
            return subhasher(p.second) ^ (subhasher(p.first) << 16);
        }
    };
}

namespace neat {

	class InnovationContainer {
	private:
		unsigned int _number = 0;
		std::unordered_map<
            std::pair<unsigned int, unsigned int>,
            unsigned int
        > track{};
		void set_innovation_number(unsigned int num){ _number = num; reset(); }

		friend class NeatPool;
	public:
		void reset(){ track.clear(); };
		unsigned int add_gene(Gene& g);
		unsigned int number() const noexcept { return _number; }
	};

}