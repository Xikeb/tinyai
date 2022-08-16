#pragma once

#include <vector>
#include <map>
#include <random>
#include <list>

#include "InnovationContainer.hpp"
#include "Gene.hpp"
#include "Genotype.hpp"
#include "Specie.hpp"
#include "MutationRateContainer.hpp"
#include "SpeciatingParameterContainer.hpp"

namespace neat {

	/* a small world, where individuals (genomes) are making babies and evolving,
	 * becoming better and better after each generation :)
	 */
	class NeatPool {
	private:
		NeatPool() = delete;

		/* important part, only accecible for friend */
		InnovationContainer innovation;

		unsigned int generation_number = 1;

		/* evolutionary methods */
		Genotype crossover(const Genotype& g1, const Genotype& g2);
		void mutate_weight(Genotype& g);
		void mutate_enable_disable(Genotype& g, bool enable);
		void mutate_link(Genotype& g, bool force_bias);
		void mutate_node(Genotype& g);
		void mutate(Genotype& g);

		bool is_same_species(const Genotype& g1, const Genotype& g2) const;

		/* Specie ranking */
		void rank_globally();
		unsigned int total_average_fitness();

		/* evolution */
		void cull_species(bool cut_to_one);
		Genotype breed_child(Specie& s);
		void remove_stale_species();
		void remove_weak_species();
		void add_to_species(Genotype& child);


	public:
		/* NeatPool parameters */
		unsigned int max_fitness = 0;

		/* mutation parameters */
		MutationRateContainer mutation_rates;

		/* species parameters */
		SpeciatingParameterContainer speciating_parameters;

		/* neural network parameters */
		NetworkInfoContainer network_info;

		// NeatPool's local random number generator
		std::random_device rd;
		std::mt19937 generator;

		/* species */
		std::list<Specie> species;

		// constructor
		NeatPool(unsigned int input, unsigned int output, unsigned int bias = 1, bool rec = false, MutationRateContainer const &mutation_rates = {}, SpeciatingParameterContainer const &speciating_parameters = {}):
        network_info{
                .input_size = input,
                .bias_size = bias,
                .output_size = output,
                .functional_nodes = input + output + bias,
                .recurrent = rec,
        },
        mutation_rates(mutation_rates),
        speciating_parameters(speciating_parameters)
        {
			// seed the mersenne twister with
			// a random number from our computer
			generator.seed(rd());

			// create a basic generation with default genomes
			for (unsigned int i = 0; i<this->speciating_parameters.population; i++){
				Genotype new_genome(this->network_info, this->mutation_rates);
				this->mutate(new_genome);
				this->add_to_species(new_genome);
			}
		}

		/* next generation */
		void new_generation();
		unsigned int generation() const { return this->generation_number; }

		/* calculate fitness */
		std::vector<std::pair<Specie*, Genotype*>> get_genomes(){
			std::vector<std::pair<Specie*, Genotype*>> genomes;

            for (auto &specie: this->species) {
                for (auto &genome: specie.genomes) {
                    genomes.emplace_back(&specie, &genome);
                }
            }
			return genomes;
		}

		/* import and export */
		void import_fromfile(const std::string& filename);
		void export_tofile(const std::string& filename);
	};
}