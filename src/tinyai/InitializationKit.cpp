#include "tinyai/Phenotypes/InitializationKit.hpp"

ann::InitializationKit::InitializationKit() {
	this->table.set_loads({
		0.f, this->table.get_loads().second
	});
	// this->evaluatedNodesSet.set_loads({
		// 0.f, this->evaluatedNodesSet.get_loads().second
	// });
	this->seenNodes.set_loads({
		0.f, this->seenNodes.get_loads().second
	});
}

void ann::InitializationKit::reset() noexcept {
	table.clear();
	nodeCount = 0;
	// evaluatedNodesSet.clear();
	evaluatedNodes.clear();
	lengthOfUnnecessaryEvaluation = 0;
	plannedNodes.clear();
	seenNodes.clear();
}

void ann::InitializationKit::indexFunctionalNodes(neat::NetworkInfoContainer const &network_info) noexcept {
	unsigned int count = network_info.input_size + network_info.bias_size + network_info.output_size;

	for (unsigned int i = 0; i < count; i++) {
		this->table.insert({i, i});
	}

	nodeCount += count;
}

bool ann::InitializationKit::indexNode(unsigned int origNodeId) noexcept {
	if (table.end() != table.find(origNodeId)) {
		return false;
	}

	table.insert({origNodeId, nodeCount});
	++nodeCount;

	return true;
}

unsigned int ann::InitializationKit::nodeIndex(unsigned int origNodeId) const noexcept {
	return table.find(origNodeId)->second;
}

void ann::InitializationKit::markFunctionalEvaluated(size_t startIdx, size_t count) noexcept {
	for(size_t i = 0; i < count; ++i) {
		markEvaluated(startIdx + i);
	}
}
void ann::InitializationKit::markEvaluated(size_t nodeId) noexcept {
	// if (this->evaluatedNodesSet.insert(nodeId).second) {
		this->evaluatedNodes.push_back(nodeId);
	// }
}

void ann::InitializationKit::addPlannedNodes(size_t first, size_t count) noexcept {
	size_t currSize = this->plannedNodes.size();

	this->plannedNodes.resize(currSize + count);
	auto beg = this->plannedNodes.begin() + currSize;

	for(size_t i = 0; i < count; ++i) {
		*(beg++) = first + i;
	}

    // this->plannedNodes.insert(
    //     this->plannedNodes.end(),
    //     first, last
    // );
}

void ann::InitializationKit::addPlannedNode(size_t nodeId) noexcept {
	this->plannedNodes.push_back(nodeId);
}
size_t ann::InitializationKit::nextPlannedNode() const noexcept {
	return this->plannedNodes.back();
}
void ann::InitializationKit::popPlannedNode() noexcept {
	this->plannedNodes.pop_back();
}
bool ann::InitializationKit::hasPlannedNodes() const noexcept {
	return not this->plannedNodes.empty();
}

bool ann::InitializationKit::hasSeenNode(size_t nodeId) const noexcept {
	return this->seenNodes.find_end() != this->seenNodes.find(nodeId);
}


