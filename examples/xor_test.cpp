#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include "../src/tinyneat.hpp"
#include "../src/tinyann.hpp"

// returns the fitness.
unsigned int xor_test(ann::neuralnet& n, bool write_output){
	std::vector<double> input(2, 0.0);
	std::vector<double> output(1, 0.0);
	unsigned int fitness = 0;
	double score;
	double answer;

	if (write_output) std::cerr << "     > begin xor test" << std::endl << "        (";

	input[0] = 0.0, input[1] = 0.0, answer = 0.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << " ";
	fitness += std::min(std::abs(0.5 / std::abs(answer - output[0])), 3.0);	

	input[0] = 0.0, input[1] = 1.0, answer = 1.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << " ";
	fitness += std::min(std::abs(0.5 / std::abs(answer - output[0])), 3.0);

	input[0] = 1.0, input[1] = 0.0, answer = 1.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << " ";
	fitness += std::min(std::abs(0.5 / std::abs(answer - output[0])), 3.0);

	input[0] = 1.0, input[1] = 1.0, answer = 0.0;
	n.evaluate(input, output);
	if (write_output) std::cerr << output[0] << ")" << std::endl;
	fitness += std::min(std::abs(0.5 / std::abs(answer - output[0])), 3.0);

	

	return fitness;
}

void test_output(){
	ann::neuralnet n;
	n.import_fromfile("fit = 8");
	xor_test(n, true);	
}

int main(){
	neat::pool p(2, 1, 1);
	srand(time(NULL));
	unsigned int max_fitness = 0;
	while (max_fitness < 12){
		unsigned int local_max_fitness = 0;
		unsigned int current_fitness = 0;		
		for (auto s = p.species.begin(); s != p.species.end(); s++)
			for (size_t i=0; i<(*s).genomes.size(); i++){				
				ann::neuralnet n;
				neat::genome& g = (*s).genomes[i];
				n.from_genome(g);
				current_fitness = xor_test(n, false);
				if (current_fitness > local_max_fitness)
					local_max_fitness = current_fitness;
				if (current_fitness > max_fitness){
					max_fitness = current_fitness;
					std::string fname = "fit = " + std::to_string(current_fitness);
					n.export_tofile(fname);
				}
				g.fitness = current_fitness;
			}

		std::cerr << "Generation " << p.generation() << " successfuly tested. Local max fitness: " << local_max_fitness << ", Global max fitness: " << max_fitness << std::endl;
		p.new_generation();
	}

	test_output();
	p.export_tofile("xor_test.res");
	return 0;
}
