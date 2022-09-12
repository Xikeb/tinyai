#include <vector>
#include <queue>
#include <iostream>
#include <fstream>

#include "tinyai/NeatPool.hpp"

namespace neat {

	/* now the evolutionary functions itself */


	Genotype NeatPool::crossover(const Genotype& g1, const Genotype& g2){
		// Make sure g1 has the higher fitness, so we will include only disjoint/excess
		// genes from the first Genotype.
		if (g2.fitness > g1.fitness)
			return crossover(g2, g1);
		Genotype child(this->network_info, this->mutation_rates);

		auto it1 = g1.genes.begin();

		// coin flip random number distributor
		std::uniform_int_distribution<int> coin_flip(1, 2);

		for (; it1 != g1.genes.end(); it1++) {

			// if innovation marks match, do the crossover, else include from the first
			// Genotype because its fitness is not smaller than the second's

			auto it2 = g2.genes.find((*it1).second.innovation_num);
			if (it2 != g2.genes.end()) {

				// do the coin flip
				int coin = coin_flip(this->generator);

			// now, after flipping the coin, we do the crossover.
			#ifdef INCLUDE_ENABLED_GENES_IF_POSSIBLE
				if (coin == 2 && (*it2).enabled)
					child.genes[(*it1).second.innovation_num] = (*it2).second;
				else
					child.genes[(*it1).second.innovation_num] = (*it1).second;
			#else
				if (coin == 2)
					child.genes[(*it1).second.innovation_num] = (*it2).second;
				else
					child.genes[(*it1).second.innovation_num] = (*it1).second;

			#endif

			} else
					// as said before, we include the disjoint Gene
					// from the first (with larger fitness) otherwise
					child.genes[(*it1).second.innovation_num] = (*it1).second;
		}

		child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);
		return child;
	}


	/* mutations */
	void NeatPool::mutate_weight(Genotype& g) {
		double step = this->mutation_rates.step_size;
		std::uniform_real_distribution<double> real_distributor(0.0, 1.0);

		for (auto & gene : g.genes){
            if (real_distributor(this->generator) < this->mutation_rates.perturb_chance) {
                gene.second.weight += real_distributor(this->generator) * step * 2.0 - step;
            } else {
                gene.second.weight = real_distributor(this->generator) * 4.0 - 2.0;
            }
		}
	}

	void NeatPool::mutate_enable_disable(Genotype& g, bool enable) {

		// that shit is safe because there's no changings in map
		// during this function
		std::vector<Gene*> v;

        for (auto &gene: g.genes) {
            if (gene.second.enabled != enable)
                v.push_back(&(gene.second));
        }

		if (v.empty())
			return ;

		std::uniform_int_distribution<int> distributor(0, v.size()-1);
		v[distributor(this->generator)]->enabled = enable;
	}

	void NeatPool::mutate_link(Genotype& g, bool force_bias){
		/* network encoding:
		 * | input nodes | bias | output nodes |
		 */
		auto is_input = [&](unsigned int node) -> bool {
				return node < this->network_info.input_size; };
		auto is_output = [&](unsigned int node) -> bool {
				return node < this->network_info.functional_nodes && node >=
					(this->network_info.input_size + this->network_info.bias_size); };
		auto is_bias = [&](unsigned int node) -> bool {
				return node < (this->network_info.input_size + this->network_info.bias_size) && node >= this->network_info.input_size; };

		std::uniform_int_distribution<unsigned int> distributor1(0, g.max_neuron-1);
		unsigned int neuron1 = distributor1(this->generator);

		std::uniform_int_distribution<unsigned int> distributor2
			(this->network_info.input_size + this->network_info.bias_size, g.max_neuron-1);
		unsigned int neuron2 = distributor2(this->generator);

		if (is_output(neuron1) && is_output(neuron2))
			return ;
		if (is_bias(neuron2))
			return ;
		if (neuron1 == neuron2 && (!force_bias))
			return ;
		if (is_output(neuron1))
			std::swap(neuron1, neuron2);

		if (force_bias){
			std::uniform_int_distribution<unsigned int> bias_choose
				(this->network_info.input_size, this->network_info.input_size + this->network_info.output_size-1);
			neuron1 = bias_choose(this->generator);
		}

		if (!g.network_info.recurrent){
			// check for recurrency using BFS
			bool has_recurrence = false;
			if (is_bias(neuron1) || is_input(neuron1))
				has_recurrence = false;
			else {

				std::queue<unsigned int> que;
				std::vector<std::vector<unsigned int>> connections(g.max_neuron);
				for (auto & gene : g.genes) {
                    connections[gene.second.from_node].push_back(gene.second.to_node);
                }
				connections[neuron1].push_back(neuron2);

				for (size_t i=0; i<connections[neuron1].size(); i++)
					que.push(connections[neuron1][i]);
					while (!que.empty()){
					unsigned int tmp = que.front();
					if (tmp == neuron1){
						has_recurrence = true;
						break;
					}
					que.pop();
					for (size_t i=0; i<connections[tmp].size(); i++)
						que.push(connections[tmp][i]);
				}
			}
			if (has_recurrence)
				return ;

			// now we are sure that it doesn't has any recurrency
		}

		// now we can create a link
		Gene new_gene;
		new_gene.from_node = neuron1;
		new_gene.to_node = neuron2;

		// if Genotype already has this connection
		for (auto & gene : g.genes)
			if (gene.second.from_node == neuron1 && gene.second.to_node == neuron2)
				return ;

		// add new innovation if needed
		new_gene.innovation_num = this->innovation.add_gene(new_gene);

		// mutate new link
		std::uniform_real_distribution<double> weight_generator(0.0, 1.0);
		new_gene.weight = weight_generator(this->generator) * 4.0 - 2.0;

		g.genes[new_gene.innovation_num] = new_gene;
	}


	void NeatPool::mutate_node(Genotype& g){
		if (g.genes.empty())
			return ;

		g.max_neuron++;

		// randomly choose a Gene to mutate
		std::uniform_int_distribution<unsigned int> distributor(0, g.genes.size()-1);
		unsigned int gene_id = distributor(this->generator);
		auto it = g.genes.begin();
		std::advance(it, gene_id);

		if (!(*it).second.enabled)
			return ;

		(*it).second.enabled = false;

		Gene new_gene1;
		new_gene1.from_node = (*it).second.from_node;
		new_gene1.to_node = g.max_neuron-1; // to the last created neuron
		new_gene1.weight = 1.0;
		new_gene1.innovation_num = this->innovation.add_gene(new_gene1);
		new_gene1.enabled = true;

		Gene new_gene2;
		new_gene2.from_node = g.max_neuron-1; // from the last created neuron
		new_gene2.to_node = (*it).second.to_node;
		new_gene2.weight = (*it).second.weight;
		new_gene2.innovation_num = this->innovation.add_gene(new_gene2);
		new_gene2.enabled = true;

		g.genes[new_gene1.innovation_num] = new_gene1;
		g.genes[new_gene2.innovation_num] = new_gene2;
	}

	void NeatPool::mutate(Genotype& g){
		double coefficient[2] = {0.95, 1.05263};

		std::uniform_int_distribution<int> coin_flip(0, 1);

		g.mutation_rates.enable_mutation_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.disable_mutation_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.connection_mutate_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.node_mutation_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.link_mutation_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.bias_mutation_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.crossover_chance *= coefficient[coin_flip(this->generator)];
		g.mutation_rates.perturb_chance *= coefficient[coin_flip(this->generator)];

		std::uniform_real_distribution<double> mutate_or_not_mutate(0.0, 1.0);

		if (mutate_or_not_mutate(this->generator) < g.mutation_rates.connection_mutate_chance)
			this->mutate_weight(g);

		double p;

	    p = g.mutation_rates.link_mutation_chance;
		while (p > 0.0) {
			if (mutate_or_not_mutate(this->generator) < p)
				this->mutate_link(g, false);
			p = p - 1.0;
		}

		p = g.mutation_rates.bias_mutation_chance;
		while (p > 0.0) {
			if (mutate_or_not_mutate(this->generator) < p)
				this->mutate_link(g, true);
			p = p - 1.0;
		}

		p = g.mutation_rates.node_mutation_chance;
		while (p > 0.0) {
			if (mutate_or_not_mutate(this->generator) < p)
				this->mutate_node(g);
			p = p - 1.0;
		}

		p = g.mutation_rates.enable_mutation_chance;;
		while (p > 0.0) {
			if (mutate_or_not_mutate(this->generator) < p)
				this->mutate_enable_disable(g, true);
			p = p - 1.0;
		}

		p = g.mutation_rates.disable_mutation_chance;
		while (p > 0.0) {
			if (mutate_or_not_mutate(this->generator) < p)
				this->mutate_enable_disable(g, false);
			p = p - 1.0;
		}

	}

	bool NeatPool::is_same_species(const Genotype& g1, const Genotype& g2) const {
		double dd = this->speciating_parameters.delta_disjoint * Genotype::disjoint(g1, g2);
		double dw = this->speciating_parameters.delta_weights * Genotype::weights(g1, g2);
		return dd + dw < this->speciating_parameters.delta_threshold;
	}

	void NeatPool::rank_globally() {
		std::vector<Genotype*> global;
        for (auto &specie: this->species) {
            for (auto &genome : specie.genomes) {
                global.push_back(&genome);
            }
        }

		std::sort(global.begin(), global.end(), [](Genotype const *a, Genotype const *b) -> bool {
            return a->fitness < b->fitness;
        });
        unsigned long genomeCount = global.size();
        for (size_t j=0; j < genomeCount; j++)
			global[j]->global_rank = j+1;
	}

	unsigned int NeatPool::total_average_fitness() {
		unsigned int total = 0;
		for (auto & specie : this->species)
			total += specie.average_fitness;
		return total;
	}


	void NeatPool::cull_species(bool cut_to_one) {
		for (auto & specie : this->species) {
			std::sort(specie.genomes.begin(), specie.genomes.end(),
					[](Genotype const &a, Genotype const &b){ return a.fitness > b.fitness; });

			unsigned int remaining = std::ceil(specie.genomes.size() * 1.0 / 2.0);
			// this will leave the most fit Genotype in Specie,
			// letting him make more and more babies (until someone in
			// Specie beat him or he becomes weaker during mutations
			if (cut_to_one)
				remaining = 1;
			while (specie.genomes.size() > remaining)
				specie.genomes.pop_back();
		}
	}

	Genotype NeatPool::breed_child(Specie &s){
		Genotype child(this->network_info, this->mutation_rates);
		std::uniform_real_distribution<double> distributor(0.0, 1.0);
		std::uniform_int_distribution<unsigned int> choose_genome(0, s.genomes.size()-1);
		if (distributor(this->generator) < this->mutation_rates.crossover_chance){
			unsigned int g1id, g2id;
			Genotype& g1 = s.genomes[g1id = choose_genome(this->generator)];
			Genotype& g2 = s.genomes[g2id = choose_genome(this->generator)];

			// QUESTION: if g1 == g2, then you can make a baby by fapping?
			child = this->crossover(g1, g2);
		} else {
			Genotype& g = s.genomes[choose_genome(this->generator)];
			child = g;
		}

		this->mutate(child);
		return child;
	}

	void NeatPool::remove_stale_species() {
		auto s = this->species.begin();
		while (s != this->species.end()) {
			Genotype& g = *(std::max_element((*s).genomes.begin(), (*s).genomes.end(),
                                             [](Genotype& a, Genotype& b) -> bool { return a.fitness < b.fitness; }));

			if (g.fitness > (*s).top_fitness){
				(*s).top_fitness = g.fitness;
				(*s).staleness = 0;
			} else {
                (*s).staleness++;
            }

			if (! ((*s).staleness < this->speciating_parameters.stale_species || (*s).top_fitness >= this->max_fitness)) {
				this->species.erase(s);
            }

            ++s;
		}
	}

	void NeatPool::remove_weak_species() {
		unsigned int sum = this->total_average_fitness();
		auto s = this->species.begin();
		while (s != this->species.end()){
			double breed = std::floor((1. * (*s).average_fitness)/(1. * sum)*1.*this->speciating_parameters.population);
			if (breed >= 1.0)
				s++;
			else
				this->species.erase(s++);
		}
	}

	void NeatPool::add_to_species(Genotype& child){
		auto s = this->species.begin();
		while (s != this->species.end()){
			if (this->is_same_species(child, (*s).genomes[0])){
				(*s).genomes.push_back(child);
				break;
			}
			++s;
		}

		if (s == this->species.end()){
			Specie new_specie;
			new_specie.genomes.push_back(child);
			this->species.push_back(new_specie);
		}
	}

	void NeatPool::new_generation() {
		this->innovation.reset();
		this->cull_species(false);
		this->rank_globally();
		this->remove_stale_species();

        for (auto &specie: this->species) {
            specie.calculate_average_fitness();
        }
		this->remove_weak_species();

		std::vector<Genotype> children;
		unsigned int sum = this->total_average_fitness();
		for (auto &specie : this->species){
			unsigned int breed =
				std::floor( ((1.*specie.average_fitness) / (1.*sum))*1.*this->speciating_parameters.population) - 1;
			for (unsigned int i = 0; i < breed; i++)
				children.emplace_back(std::move(this->breed_child(specie)));
		}

		this->cull_species(true); // now in each species we have only one Genotype

		// preparing for MAKING BABIES <3
		std::uniform_int_distribution<unsigned int> choose_specie(0, this->species.size()-1);
		std::vector<Specie*> species_pointer;
        species_pointer.reserve(this->species.size());
		for (auto & specie : this->species)
			species_pointer.push_back(&specie);
        if (this->species.empty()) {
            std::cerr << "Wtf? Zero species in the world! All dead? Where is that fucking NOAH and his fucking boat?\n";
        } else {
            while (children.size() + this->species.size() < this->speciating_parameters.population) {
                children.push_back(this->breed_child(*species_pointer[choose_specie(this->generator)]));
            }
        }

		for (auto &child: children)
			this->add_to_species(child);
		this->generation_number++;
	}

	void NeatPool::import_fromfile(const std::string& filename) {
		std::ifstream input;
		input.open(filename);
		if (!input.is_open()){
			std::cerr << "cannot open file '" << filename << "' !";
			return ;
		}

		this->species.clear();
		try {
			// current state
			unsigned int innovation_num;
			input >> innovation_num;
			this->innovation.set_innovation_number(innovation_num);
			input >> this->generation_number;
			input >> this->max_fitness;

			// network information
			input >> this->network_info.input_size >> this->network_info.output_size >> this->network_info.bias_size;
			this->network_info.functional_nodes = this->network_info.input_size +
				this->network_info.output_size + this->network_info.bias_size;

			std::string rec;
			input >> rec;
			if (rec == "rec")
				this->network_info.recurrent = true;
			if (rec == "nonrec")
				this->network_info.recurrent = false;

			// population information
			this->speciating_parameters.read(input);

			// mutation parameters
			this->mutation_rates.read(input);

			// species information
			unsigned int species_number;
			input >> species_number;
			this->species.clear();

			for (unsigned int c = 0; c < species_number; c++){
				Specie new_specie;
			#ifdef GIVING_NAMES_FOR_SPECIES
				input >> new_specie.name;
			#endif
				input >> new_specie.top_fitness;
				input >> new_specie.average_fitness;
				input >> new_specie.staleness;

				unsigned int specie_population;
				input >> specie_population;

				for (unsigned int i=0; i<specie_population; i++) {
					Genotype new_genome(this->network_info, this->mutation_rates);
					input >> new_genome.fitness;
					input >> new_genome.adjusted_fitness;
					input >> new_genome.global_rank;

					new_genome.mutation_rates.read(input);

					unsigned int gene_number;
					input >> new_genome.max_neuron >> gene_number;

					for (unsigned int j=0; j<gene_number; j++){
						Gene new_gene;
						input >> new_gene.innovation_num;
						input >> new_gene.from_node;
						input >> new_gene.to_node;
						input >> new_gene.weight;
						input >> new_gene.enabled;
						new_genome.genes[new_gene.innovation_num] = new_gene;
					}

					new_specie.genomes.push_back(new_genome);
				}

				this->species.push_back(new_specie);
			}

		} catch (std::string const &error_message){
			std::cerr << error_message;
		}

		input.close();
	}

	void NeatPool::export_tofile(const std::string& filename){
		std::ofstream output;
		output.open(filename);
		if (!output.is_open()){
			std::cerr << "cannot open file '" << filename << "' !";
			return ;
		}

		// current state
		output << this->innovation.number() << std::endl;
		output << this->generation_number << std::endl;
		output << this->max_fitness << std::endl;

		// network information
		output << this->network_info.input_size << " " <<  this->network_info.output_size << " " <<
		   	this->network_info.bias_size << std::endl;
		if (this->network_info.recurrent)
			output << "rec" << std::endl;
		else
			output << "nonrec" << std::endl;
		this->network_info.functional_nodes = this->network_info.input_size +
		   	this->network_info.output_size + this->network_info.bias_size;

		// population information
		this->speciating_parameters.write(output, "");

		// mutation parameters
		this->mutation_rates.write(output, "");

		// species information
		output << this->species.size() << std::endl;
		for (auto & specie : this->species){
			output << "   ";
		#ifdef GIVING_NAMES_FOR_SPECIES
			output << (*s).name << " ";
		#endif
			output << specie.top_fitness << " ";
			output << specie.average_fitness << " ";
			output << specie.staleness << std::endl;

			output << "   " << specie.genomes.size() << std::endl;
			for (auto & genome : specie.genomes) {
				output << "      ";
			    output << genome.fitness << " ";
				output << genome.adjusted_fitness << " ";
				output << genome.global_rank << std::endl;

				genome.mutation_rates.write(output, "      ");

				output << "      " << genome.max_neuron << " " <<
				   	genome.genes.size() << std::endl;
				for (auto & it : genome.genes) {
					Gene& g = it.second;
					output << "         ";
					output << g.innovation_num << " " << g.from_node << " " << g.to_node << " "
						<< g.weight << " " << g.enabled << std::endl;
				}
			}

			output << std::endl << std::endl;
		}
		output.close();
	}

} // end of namespace neat