//
// Created by eliord on 7/24/22.
//

#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <cstdlib>

#include "tinyai/Phenotypes/IPhenotype.hpp"
#include "tinyai/Phenotypes/InitializationKit.hpp"

namespace ann {

    /**
     * Serial execution of neurons
     */
    class NeuralNetFastSerialRecurrent {
    public:
        //NodeId => connectionWeight
        using Synapse = std::pair<std::size_t, double>;
        friend InitializationKit;

    private:
        std::vector<double> nodeActivation;
        std::vector<long long unsigned int> nodeActivated;
        bool allActivated = false;
        std::unordered_multimap<size_t, Synapse> links;

        std::size_t input_offset, input_count;
        std::size_t bias_offset, bias_count;
        std::size_t output_offset, output_count;
        std::size_t hidden_offset, hidden_count;
        std::size_t node_count;

        bool isNodeActivated(std::size_t nodeId) const noexcept;
        void setNodeActivated(std::size_t nodeId, bool value = true) noexcept;
        bool outputsAreAllActivated() const noexcept;

        void evaluate_dirty() noexcept;
        void evaluate_clean() noexcept;

    public:
        constexpr static bool isRecurrent() noexcept {
            return false;
        }

        void evaluate(const std::vector<double>& input, std::vector<double>& output);
        void from_genome(neat::Genotype const &a, ann::InitializationKit &initKit);
    };

} // ann
