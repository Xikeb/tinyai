#pragma once

#include <vector>
#include <stack>

#include "tinyai/detail.hpp"
#include "tinyai/NetworkInfoContainer.hpp"

namespace ann {
	struct InitializationKit {
        //NodeId => connectionWeight
        using Synapse = std::pair<std::size_t, double>;

        gnu_hash_table<unsigned int, unsigned int> table;
	    size_t nodeCount;
        std::unordered_multimap<size_t, Synapse> links;

	    //the set tells me if the node is known, the vector holds them in order
	    // gnu_hash_set<size_t> evaluatedNodesSet;
	    std::vector<size_t> evaluatedNodes;

	    size_t lengthOfUnnecessaryEvaluation;
	    std::vector<std::size_t> plannedNodes;
	    gnu_hash_set<size_t> seenNodes;

        InitializationKit();

	    void reset() noexcept;

	    void indexFunctionalNodes(neat::NetworkInfoContainer const &network_info) noexcept;
	    bool indexNode(unsigned int origNodeId) noexcept;
	    unsigned int nodeIndex(unsigned int origNodeId) const noexcept;

	    template<typename InputIt>
	    void markEvaluated(InputIt first, InputIt last) noexcept;

        void markFunctionalEvaluated(size_t startIdx, size_t count) noexcept;

        void markEvaluated(size_t nodeId) noexcept;

        auto startOfEvaluation() const noexcept {
        	auto start = this->evaluatedNodes.cbegin();

        	std::advance(start, lengthOfUnnecessaryEvaluation);

        	return start;
        }

        auto endOfEvaluation() const noexcept {
        	return this->evaluatedNodes.cend();
        }

	    void addPlannedNodes(size_t first, size_t count) noexcept;
	    void addPlannedNode(size_t nodeId) noexcept;
	    size_t nextPlannedNode() const noexcept;
	    void popPlannedNode() noexcept;
	    bool hasPlannedNodes() const noexcept;
	    bool hasSeenNode(size_t nodeId) const noexcept;

	    template<typename InputIt>
	    void addSeenNodes(InputIt first, InputIt last) noexcept {
            seenNodes.copy_from_range(first, last);
            // for(auto it = first; it != last; ++it) {
            // 	this->seenNodes.insert(*it);
            // }
            // this->seenNodes.insert(first, last);
	    }

	    bool addSeenNode(size_t nodeId) noexcept {
	        return this->seenNodes.insert(nodeId).second;
	    }
	};

} // ann