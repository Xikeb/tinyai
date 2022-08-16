//
// Created by eliord on 4/10/22.
//

#include <unordered_set>
#include <set>
#include <iterator>
#include <numeric>
#include <algorithm>

#pragma GCC optimize("Ofast,-funroll-loops,-march=native,-ffast-math,-ffinite-math-only,-funroll-loops,-fomit-frame-pointer")
//#pragma GCC optimize("O3,-funroll-loops,-march=native,-funroll-loops,-fomit-frame-pointer")

#include "NeuralNetFast.hpp"

void ann::NeuralNetFast::evaluate_nonrecurrent(std::vector<double> const &input, std::vector<double> &output) {
    for (auto &activation: nodeActivation) {
        activation = 0;
    }

    auto inputEnd = std::min(input.end(), input.begin() + input_count);
    std::copy(input.begin(), inputEnd, nodeActivation.begin());

    auto biasBegin = std::min(nodeActivation.begin() + bias_offset, nodeActivation.end());
    auto biasEnd = std::min(biasBegin + bias_count, nodeActivation.end());
    std::fill(biasBegin, biasEnd, 1.0);

    for(auto nodeId: evaluationOrder) {
        double excitation = 0.;

        for(auto [iter, itEnd] = links.equal_range(nodeId); iter != itEnd; ++iter) {
            auto const &[inNodeId, connectionWeight] = iter->second;
            excitation += nodeActivation.at(inNodeId) * connectionWeight;
        }

        nodeActivation.at(nodeId) = sigmoid(excitation);
    }

    std::copy(
            nodeActivation.begin() + output_offset,
            nodeActivation.begin() + output_offset + output_count,
            std::begin(output)
    );
}

void ann::NeuralNetFast::evaluate_recurrent(std::vector<double> const &input, std::vector<double> &output) {
    auto inputEnd = std::min(input.end(), input.begin() + input_count);
    std::copy(input.begin(), inputEnd, nodeActivation.begin());

    auto biasBegin = std::min(nodeActivation.begin() + bias_offset, nodeActivation.end());
    auto biasEnd = std::min(biasBegin + bias_count, nodeActivation.end());
    std::fill(biasBegin, biasEnd, 1.0);

    // in non-recurrent, each node we will visit only one time per
    // simulation step (similar to the real world)
    // and the values will be saved till the next simulation step
    for(auto nodeId: evaluationOrder) {
        double excitation = 0.;

        for(auto [iter, itEnd] = links.equal_range(nodeId); iter != itEnd; ++iter) {
            auto const &[inNodeId, connectionWeight] = iter->second;
            excitation += nodeActivation.at(inNodeId) * connectionWeight;
        }

        nodeActivation.at(nodeId) = sigmoid(excitation);
    }

    std::copy(
            nodeActivation.begin() + output_offset,
            nodeActivation.begin() + output_offset + output_count,
            std::begin(output)
    );
}

void ann::NeuralNetFast::from_genome(neat::Genotype const &a, ann::InitializationKit &initKit) {
    unsigned int input_size = a.network_info.input_size;
    unsigned int output_size = a.network_info.output_size;
    unsigned int bias_size = a.network_info.bias_size;

    this->recurrent = a.network_info.recurrent;

    nodeActivation.clear();
    links.clear();
    evaluationOrder.clear();

    size_t nodeId = 0;

    input_count = input_size; input_offset = nodeId;
    nodeId += input_size;
    bias_count = bias_size; bias_offset = nodeId;
    nodeId += bias_size;
    output_count = output_size; output_offset = nodeId;
    nodeId += output_size;

    initKit.reset();

    this->links.reserve(a.genes.size());
    // initKit.links.reserve(a.genes.size());
    initKit.table.resize(input_size + bias_size + output_size + a.genes.size() * 2);
    initKit.indexFunctionalNodes(a.network_info);

    for (const auto & [_, gene] : a.genes) {
        if (!gene.enabled)
            continue;

        initKit.indexNode(gene.from_node);
        initKit.indexNode(gene.to_node);
    }

    nodeActivation.resize(initKit.nodeCount);
    initKit.evaluatedNodes.reserve(initKit.nodeCount);
    // initKit.evaluatedNodesSet.resize(initKit.nodeCount);
    initKit.seenNodes.resize(initKit.nodeCount);

    for (auto const &[_, gene] : a.genes) {
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

            for(auto [iter, itEnd] = this->links.equal_range(nodeId); iter != itEnd; ++iter) {
                auto const [inNodeId, _] = iter->second;
                if (not initKit.hasSeenNode(inNodeId)) {
                    // if we haven't calculated activation for this node
                    initKit.addPlannedNode(inNodeId);
                };
            }
        }
    }

    this->evaluationOrder.insert(
            this->evaluationOrder.end(),
            initKit.startOfEvaluation(), initKit.endOfEvaluation()
    );
}


void ann::NeuralNetFast::from_genome(neat::Genotype const &a) {
    unsigned int input_size = a.network_info.input_size;
    unsigned int output_size = a.network_info.output_size;
    unsigned int bias_size = a.network_info.bias_size;


    this->recurrent = a.network_info.recurrent;

    nodeActivation.clear();
    links.clear();
    evaluationOrder.clear();

    std::unordered_map<unsigned int, unsigned int> table;
    for (unsigned int i = 0; i < input_size + bias_size + output_size; i++) {
        table.insert({i, i});
    }

    size_t nodeCount = 0;

    input_offset = nodeCount; input_count = input_size;
    nodeCount += input_size;

    bias_offset = nodeCount; bias_count = bias_size;
    nodeCount += bias_size;
    
    output_offset = nodeCount; output_count = output_size;
    nodeCount += output_size;

    for (const auto & [_, gene] : a.genes) {
        if (!gene.enabled)
            continue;

        nodeCount += static_cast<size_t>(table.try_emplace(gene.from_node, nodeCount).second);
        nodeCount += static_cast<size_t>(table.try_emplace(gene.to_node, nodeCount).second);
    }

    nodeActivation.resize(nodeCount);


    for (auto const &[_, gene] : a.genes) {
        if (!gene.enabled)
            continue;

        links.emplace(table.at(gene.to_node), Synapse{table.at(gene.from_node), gene.weight});
    }


    //Needs to be ordered, nodes appear here in evaluation order
    std::vector<size_t> evaluatedNodes;
    std::stack<size_t> plannedNodes{};
    //Those nodes do not need to have their excitation/activation evaluated dynamically


    for(size_t i = 0; i < input_count; ++i) {
        evaluatedNodes.push_back({input_offset + i});
    }
    for(size_t i = 0; i < bias_count; ++i) {
        evaluatedNodes.push_back({bias_offset + i});
    }

    for(size_t i = 0; i < output_count; ++i) {
        plannedNodes.push({output_offset + i});
    }

    size_t lengthOfUnnecessaryEvaluation = evaluatedNodes.size();
    std::unordered_set<size_t> seenNodes(evaluatedNodes.cbegin(), evaluatedNodes.cend());

    while (not plannedNodes.empty()) {
        size_t nodeId = plannedNodes.top();

        if (seenNodes.contains(nodeId)) {
            evaluatedNodes.push_back({nodeId});
            plannedNodes.pop();
        } else {
            seenNodes.insert(nodeId);

            for(auto [iter, itEnd] = links.equal_range(nodeId); iter != itEnd; ++iter) {
                auto const [inNodeId, _] = iter->second;
                if (not seenNodes.contains(inNodeId)) {
                    // if we haven't calculated activation for this node
                    plannedNodes.push(inNodeId);
                };
            }
        }
    }

    auto startOfEvaluation = evaluatedNodes.cbegin();
    std::advance(startOfEvaluation, lengthOfUnnecessaryEvaluation);

    evaluationOrder.insert(
            evaluationOrder.end(),
            startOfEvaluation, evaluatedNodes.cend()
    );
}

void ann::NeuralNetFast::evaluate(std::vector<double> const &input, std::vector<double> &output) {
    if (recurrent)
        this->evaluate_recurrent(input, output);
    else
        this->evaluate_nonrecurrent(input, output);
}
