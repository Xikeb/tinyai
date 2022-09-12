//
// Created by eliord on 7/24/22.
//

#include <algorithm>

#include "tinyai/Phenotypes/NeuralNetFastSerialRecurrent.hpp"
#include "tinyai/Phenotypes/InitializationKit.hpp"

namespace ann {
    void NeuralNetFastSerialRecurrent::evaluate_dirty() noexcept {
        for(auto nodeId = 0; nodeId < input_count; ++nodeId) {
            setNodeActivated(input_offset + nodeId);
        }
        for(auto nodeId = 0; nodeId < bias_count; ++nodeId) {
            setNodeActivated(bias_offset + nodeId);
        }

        do{
            for (std::size_t nodeId = bias_offset + bias_count; nodeId < node_count; ++nodeId) {
                auto [iter, itEnd] = links.equal_range(nodeId);

                if (iter == itEnd) {
                    setNodeActivated(nodeId);
                    continue;
                }

                bool anyIsActivated = false;
                double excitation = 0.;

                for (; iter != itEnd; ++iter) {
                    auto const &[inNodeId, connectionWeight] = iter->second;
                    excitation += nodeActivation.at(inNodeId) * connectionWeight;
                    anyIsActivated = anyIsActivated || isNodeActivated(nodeId);
                }

                setNodeActivated(nodeId, anyIsActivated);
                nodeActivation.at(nodeId) = sigmoid(excitation);
            }
        } while(!outputsAreAllActivated());

        allActivated = true;
    }

    void NeuralNetFastSerialRecurrent::evaluate_clean() noexcept {
        constexpr static auto compareFirst = [](const std::pair<size_t, Synapse> & lhs, const std::pair<size_t, Synapse> & rhs) -> bool {
            return lhs.first < rhs.first;
        };
        auto it = links.cbegin();
        auto end = links.cend();

        while(it != end) {
            double excitation = 0.;

            for (auto next = std::upper_bound(it, links.cend(), *it, compareFirst); it != next; ++it) {
                auto const &[inNodeId, connectionWeight] = it->second;
                excitation += nodeActivation.at(inNodeId) * connectionWeight;
            }

            auto nodeId = it->first;
            nodeActivation.at(nodeId) = sigmoid(excitation);
        }
    }

    void NeuralNetFastSerialRecurrent::evaluate(std::vector<double> const &input, std::vector<double> &output) {
        std::copy(input.begin(), input.begin() + input_count, nodeActivation.begin());

        auto biasBegin = nodeActivation.begin() + bias_offset;
        std::fill(biasBegin, biasBegin + bias_count, 1.0);

        if (!allActivated) {
            evaluate_dirty();
        } else {
            evaluate_clean();
        }

        auto const outputBegin = nodeActivation.begin() + output_offset;
        std::copy(outputBegin, outputBegin + output_count, std::begin(output));
    }


    void NeuralNetFastSerialRecurrent::from_genome(neat::Genotype const &a, ann::InitializationKit &initKit) {
        unsigned int input_size = a.network_info.input_size;
        unsigned int output_size = a.network_info.output_size;
        unsigned int bias_size = a.network_info.bias_size;

        nodeActivation.clear();
        links.clear();

        size_t nodeId = 0;

        input_count = input_size; input_offset = nodeId;
        nodeId += input_size;
        bias_count = bias_size; bias_offset = nodeId;
        nodeId += bias_size;
        output_count = output_size; output_offset = nodeId;
        nodeId += output_size;

        initKit.reset();

        this->links.reserve(a.genes.size());
        initKit.table.resize(input_size + bias_size + output_size + a.genes.size() * 2);
        initKit.indexFunctionalNodes(a.network_info);

        for (const auto &[_, gene]: a.genes) {
            if (!gene.enabled)
                continue;

            initKit.indexNode(gene.from_node);
            initKit.indexNode(gene.to_node);
        }

        hidden_offset = output_offset + output_count;
        hidden_count = initKit.nodeCount - (input_count - bias_count - output_count);
        node_count = initKit.nodeCount;
        allActivated = false;

        nodeActivation.resize(initKit.nodeCount);
        nodeActivated.resize(1 + initKit.nodeCount / (8 * sizeof(typename decltype(nodeActivation)::value_type)));
        std::fill(std::begin(nodeActivated), std::end(nodeActivated), 0);

        initKit.evaluatedNodes.reserve(initKit.nodeCount);
        // initKit.evaluatedNodesSet.resize(initKit.nodeCount);
        initKit.seenNodes.resize(initKit.nodeCount);

        for (auto const &[_, gene]: a.genes) {
            if (!gene.enabled)
                continue;

            this->links.insert({
                initKit.nodeIndex(gene.to_node),
                Synapse{initKit.nodeIndex(gene.from_node), gene.weight}
            });
        }

        initKit.markFunctionalEvaluated(input_offset, input_count);
        initKit.markFunctionalEvaluated(bias_offset, bias_count);

        initKit.lengthOfUnnecessaryEvaluation = initKit.evaluatedNodes.size();

        initKit.addPlannedNodes(output_offset, output_count);
        initKit.addSeenNodes(initKit.evaluatedNodes.cbegin(), initKit.evaluatedNodes.cend());

        while (initKit.hasPlannedNodes()) {
            size_t nodeId = initKit.nextPlannedNode();

            if (initKit.hasSeenNode(nodeId)) {
                initKit.markEvaluated(nodeId);
                initKit.popPlannedNode();
            } else {
                initKit.addSeenNode(nodeId);

                for (auto [iter, itEnd] = this->links.equal_range(nodeId); iter != itEnd; ++iter) {
                    auto const [inNodeId, _] = iter->second;
                    if (not initKit.hasSeenNode(inNodeId)) {
                        // if we haven't calculated activation for this node
                        initKit.addPlannedNode(inNodeId);
                    };
                }
            }
        }
    }

    bool NeuralNetFastSerialRecurrent::outputsAreAllActivated() const noexcept {
        using WordType = typename decltype(nodeActivated)::value_type;
        constexpr static std::size_t wordSize = 8 * sizeof(WordType);
        std::size_t beg = output_offset;
        std::size_t end = output_offset + output_count;

        if (beg % wordSize) {
            WordType ignoreMask = ~((wordSize - (beg % wordSize)) - 1);
            size_t wordIdx = beg / wordSize;

            if (~(nodeActivated.at(wordIdx) | ignoreMask)) {
                return false;
            }

            beg = (wordIdx + 1) * wordSize;
        }

        for(; beg < end; beg += wordSize) {
            if (~nodeActivated.at(beg / wordSize)) {
                return false;
            }
        }

        return true;
    }

    void NeuralNetFastSerialRecurrent::setNodeActivated(std::size_t nodeId, bool value) noexcept {
        using WordType = typename decltype(nodeActivated)::value_type;
        constexpr static std::size_t wordSize = 8 * sizeof(WordType);

        nodeActivated[nodeId / wordSize] |= (WordType(value) << (nodeId % wordSize));
    }

    bool NeuralNetFastSerialRecurrent::isNodeActivated(std::size_t nodeId) const noexcept {
        using WordType = typename decltype(nodeActivated)::value_type;
        constexpr static std::size_t wordSize = 8 * sizeof(WordType);

        return 0 != (nodeActivated.at(nodeId / wordSize) & (WordType(1) << (nodeId % wordSize)));
    }

} // ann