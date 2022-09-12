//
// Created by eliord on 7/24/22.
//

#include <fstream>
#include <iostream>

#include "tinyai/Phenotypes/NeuralNetNaiveRecurrent.hpp"
#include "tinyai/Phenotypes/IPhenotype.hpp"

namespace ann {

    void NeuralNetNaiveRecurrent::from_genome(const neat::Genotype &a) {
        unsigned int input_size = a.network_info.input_size;
        unsigned int output_size = a.network_info.output_size;
        unsigned int bias_size = a.network_info.bias_size;


        this->recurrent = a.network_info.recurrent;

        nodes.clear();
        nodes.reserve(a.network_info.functional_nodes);
        input_nodes.clear();
        input_nodes.reserve(input_size);
        bias_nodes.clear();
        bias_nodes.reserve(bias_size);
        output_nodes.clear();
        output_nodes.reserve(output_size);

        Neuron tmp;
        for (unsigned int i = 0; i < input_size; i++) {
            nodes.emplace_back(Neuron{.type = 1});
            // nodes.push_back(tmp);
            // nodes.back().type = 1;
            this->input_nodes.push_back(nodes.size() - 1);
        }
        for (unsigned int i = 0; i < bias_size; i++) {
            nodes.push_back(tmp);
            nodes.back().type = 3;
            this->bias_nodes.push_back(nodes.size() - 1);
        }
        for (unsigned int i = 0; i < output_size; i++) {
            nodes.push_back(tmp);
            nodes.back().type = 2;
            this->output_nodes.push_back(nodes.size() - 1);
        }

        std::unordered_map<unsigned int, unsigned int> table;
        for (unsigned int i = 0;
             i < input_nodes.size() + output_nodes.size() + bias_nodes.size(); i++)
            table[i] = i;

        for (const auto &[_, gene]: a.genes) {
            if (!gene.enabled)
                continue;

            if (!table.contains(gene.from_node)) {
                nodes.emplace_back();
                table[gene.from_node] = nodes.size() - 1;
            }
            if (!table.contains(gene.to_node)) {
                nodes.emplace_back();
                table[gene.to_node] = nodes.size() - 1;
            }
        }

        for (const auto &[_, gene]: a.genes) {
            nodes[table[gene.to_node]].in_nodes.emplace_back(table[gene.from_node], gene.weight);
        }
    }

    void NeuralNetNaiveRecurrent::evaluate(const std::vector<double> &input, std::vector<double> &output) {

        for (size_t i = 0; i < input.size() && i < input_nodes.size(); i++) {
            nodes[input_nodes[i]].value = input[i];
            nodes[input_nodes[i]].visited = true;
        }

        for (unsigned long bias_node: bias_nodes) {
            nodes[bias_node].value = 1.0;
            nodes[bias_node].visited = true;
        }

        // in non-recurrent, each node we will visit only one time per
        // simulation step (similar to the real world)
        // and the values will be saved till the next simulation step
        for (auto &thisNeuron: nodes) {
            double sum = 0.0;
            for (auto const [inNodeId, connectionWeight]: thisNeuron.in_nodes)
                sum += nodes[inNodeId].value + connectionWeight;
            if (!thisNeuron.in_nodes.empty())
                thisNeuron.value = ann::sigmoid(sum);
        }

        for (size_t i = 0; i < output_nodes.size() && i < output.size(); i++)
            output[i] = nodes[output_nodes[i]].value;
    }

    void NeuralNetNaiveRecurrent::import_fromfile(std::string const &filename) {
        std::ifstream o;
        o.open(filename);

        this->nodes.clear();
        this->input_nodes.clear();
        this->output_nodes.clear();

        try {
            if (!o.is_open())
                throw "error: cannot open file!";

            std::string rec;
            o >> rec;
            if (rec == "recurrent") {
                if (true != this->isRecurrent()) {
                    throw "Bad network structure.";
                }
            }
            if (rec == "non_recurrent") {
                if (false != this->isRecurrent()) {
                    throw "Bad network structure.";
                }
            }

            unsigned int neuron_number;
            o >> neuron_number;
            this->nodes.resize(neuron_number);

            for (unsigned int i = 0; i < neuron_number; i++) {
                unsigned int input_size, type; // 0 = ordinal, 1 = input, 2 = output
                nodes[i].value = 0.0;
                nodes[i].visited = false;

                o >> type;
                switch (type) {
                    case 1:
                        input_nodes.push_back(i);
                        break;
                    case 2:
                        output_nodes.push_back(i);
                        break;
                    case 3:
                        bias_nodes.push_back(i);
                        break;
                    default:
                        break;
                }
                // if (type == 1)
                // 	input_nodes.push_back(i);
                // if (type == 2)
                // 	output_nodes.push_back(i);
                // if (type == 3)
                // 	bias_nodes.push_back(i);

                nodes[i].type = type;

                o >> input_size;
                for (unsigned int j = 0; j < input_size; j++) {
                    unsigned int t;
                    double w;
                    o >> t >> w;
                    nodes[i].in_nodes.emplace_back(t, w);
                }
            }
        }
        catch (std::string const &error_message) {
            std::cerr << error_message << std::endl;
        }

        o.close();
    }

    void NeuralNetNaiveRecurrent::export_tofile(std::string const &filename) {
        std::ofstream o;
        o.open(filename);

        if (this->isRecurrent())
            o << "recurrent" << std::endl;
        else
            o << "non-recurrent" << std::endl;
        o << nodes.size() << std::endl << std::endl;

        for (size_t i = 0; i < nodes.size(); i++) {
            o << nodes[i].type << " ";
            o << nodes[i].in_nodes.size() << std::endl;
            for (unsigned int j = 0; j < nodes[i].in_nodes.size(); j++)
                o << nodes[i].in_nodes[j].first << " "
                  << nodes[i].in_nodes[j].second << " ";
            o << std::endl << std::endl;
        }
        o.close();
    }
} // ann