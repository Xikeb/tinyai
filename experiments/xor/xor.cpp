#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>

#if 0

#include <tinyneat.hpp>
#include <tinyann.hpp>

// returns the fitness.
unsigned int xor_test(ann::neuralnet& n, bool write_output){
	std::vector<double> input(2, 0.0);
	std::vector<double> output(1, 0.0);
	unsigned int fitness = 0;
	double answer;

	if (write_output) std::cerr << "     > begin xor test" << std::endl << "        (";

	input[0] = 0.0, input[1] = 0.0, answer = 0.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << " ";
	fitness += std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);	

	input[0] = 0.0, input[1] = 1.0, answer = 1.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << " ";
	fitness += std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);	

	input[0] = 1.0, input[1] = 0.0, answer = 1.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << " ";
	fitness += std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);	

	input[0] = 1.0, input[1] = 1.0, answer = 0.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << ")";
	fitness += std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);	

	if (write_output) std::cerr << " " << fitness << std::endl;

	return fitness;
}


void test_output(){
	ann::neuralnet n;
	n.import_fromfile("fit = 200");
	xor_test(n, true);	
}

int main(){
	neat::pool p(2, 1, 0, false);
	p.import_fromfile("xor_test.res");	
	srand(time(NULL));
	unsigned int max_fitness = 0;
	while (max_fitness < 200){
		unsigned int current_fitness = 0;
		unsigned int min_fitness = 100000;	
		for (auto s = p.species.begin(); s != p.species.end(); s++)
			for (size_t i=0; i<(*s).genomes.size(); i++){				
				ann::neuralnet n;
				neat::genome& g = (*s).genomes[i];
				n.from_genome(g);
				current_fitness = xor_test(n, false);
				if (current_fitness < min_fitness)
					min_fitness = current_fitness;
				if (current_fitness > max_fitness){
					max_fitness = current_fitness;
					std::string fname = "fit = " + std::to_string(current_fitness);
					n.export_tofile(fname);
				}
				g.fitness = current_fitness;
			}

		std::cerr << "Generation " << p.generation() << " successfuly tested. Global min fitness: " << min_fitness << ", Global max fitness: " << max_fitness << std::endl;
		p.new_generation();
	}

	test_output();
	p.export_tofile("xor_test.res");
	return 0;
}

#endif


#include "NeatPool.hpp"
#include "NeuralNet.hpp"
#include "NeuralNetFast.hpp"

template<typename RandomGenerator, size_t N = 1000>
unsigned int tryout(ann::NeuralNetFast &nn, RandomGenerator &rng) {
	constexpr static auto send = [](bool x) -> double {
		return x ? 1 : -1;
	};
	constexpr static auto read = [](double x) -> bool {
		return x > 0;
	};

	unsigned int score = 0;
	std::vector<double> inputs(2, 0.);
	std::vector<double> outputs(1, 0.);

	for (size_t i = N; 0 < i; --i) {
		bool a = rng();
		bool b = rng();

		inputs[0] = send(a);
		inputs[1] = send(b);

		nn.evaluate(inputs, outputs);

		score += (!a != !b) == read(outputs[0]);
	}

	return score;
}

#define UNLIKELY(x) __builtin_expect((x), 0)

template <typename U = uint64_t> class Randomizer {
  public:
    template <typename Rng> bool operator()(Rng &rng) {
        if (UNLIKELY(1 == m_rand)) {
            m_rand = std::uniform_int_distribution<U>{}(rng) | s_mask_left1;
        }
        bool const ret = m_rand & 1;
        m_rand >>= 1;
        return ret;
    }

  private:
    static constexpr const U s_mask_left1 = U(1) << (sizeof(U) * 8 - 1);
    U m_rand = 1;
};


int main() {
	neat::MutationRateContainer mutation_rates = {};
	neat::SpeciatingParameterContainer speciating_parameters = {};

	// speciating_parameters.population = 1000;
	speciating_parameters.stale_species = 5;

	// speciating_parameters.delta_disjoint = 1.0;
	// speciating_parameters.delta_weights = 0.4;
	// speciating_parameters.delta_threshold = 3.0;

	mutation_rates.connection_mutate_chance = 0.8;
	mutation_rates.perturb_chance = 0.9;
	// mutation_rates.crossover_chance = 0.95;
	// mutation_rates.disable_mutation_chance = 0.75;
	// mutation_rates.node_mutation_chance = 0.03;
	// mutation_rates.link_mutation_chance = 0.05;
	// mutation_rates.link_mutation_chance = 0.25;


	neat::NeatPool pool(
		2, 1, 1, false
		, mutation_rates, speciating_parameters
	);

	std::mt19937 generator{std::random_device{}()};
	// auto rng = [&generator, distributor = std::uniform_int_distribution<int>(1, 2)]() -> bool {
	// 	return 2 == distributor(generator);
	// };
	auto rng = [&generator, spread = Randomizer()]() mutable -> bool { return spread(generator); };

	for(unsigned int best_fitness = 0; best_fitness < 990; pool.new_generation()) {
		unsigned int current_fitness = 0;
		unsigned int max_fitness = 0;
		unsigned int min_fitness = 100000;

		for(auto &specie: pool.species) {
			for(auto &genome: specie.genomes) {
				ann::NeuralNetFast n;

				n.from_genome(genome);
				genome.fitness = current_fitness = tryout(n, rng);

				min_fitness = (current_fitness < min_fitness) ? current_fitness : min_fitness;
				max_fitness = (max_fitness < current_fitness) ? current_fitness : max_fitness;

				if (best_fitness < current_fitness) {
					best_fitness = current_fitness;
					// std::cerr << "Saving NeuralNet with HighScore=" << current_fitness << std::endl;
					// n.export_tofile("HighScore.neat.nn");
				}

			}
		}

		std::cerr << "Generation#" << pool.generation() << ": "
			<< "RecordHigh=" << best_fitness
			<< ", " << "Vanguard=" << max_fitness
			<< ", " << "Slacker=" << min_fitness
			<< std::endl;

		if (30 < pool.generation()) {
			std::cerr << "Failed to converge under 30 generations, quitting" << std::endl;

			return 1;
		}
	}

	return 0;
}