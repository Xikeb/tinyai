#include <vector>
#include <array>
#include <cmath>
#include <iostream>

#include "NeatPool.hpp"
#include "NeuralNet.hpp"
#include "NeuralNetFast.hpp"

constexpr static unsigned int maxSteps = 30*60*100;

namespace experiments::cart_pole_single {
	namespace rk {
		template<typename T>
		struct FourthOrder {
			using Self = FourthOrder<T>;

			template<typename GetAcceleration, typename ...Args>
			Self sample(GetAcceleration &&readCurve, float dt, Args&& ...args) const noexcept {
				Self next {
					.val = val + slope * dt,
					.slope = slope + curve * dt,
					.curve = curve
				};

				return Self{
					.val = val,
					.slope = next.slope,
					.curve = std::forward<GetAcceleration>(readCurve)(
						next, std::forward<Args>(args)...
					)
				};
			}

			template<typename GetAcceleration, typename ...Args>
			[[nodiscard]]
			Self step(double dt, GetAcceleration &&reader, Args&& ...args) const noexcept {
				Self k1 = this->sample(std::forward<GetAcceleration>(reader),     0, std::forward<Args>(args)...);
				Self k2 =    k1.sample(std::forward<GetAcceleration>(reader), dt/2., std::forward<Args>(args)...);
				Self k3 =    k2.sample(std::forward<GetAcceleration>(reader), dt/2., std::forward<Args>(args)...);
				Self k4 =    k3.sample(std::forward<GetAcceleration>(reader),    dt, std::forward<Args>(args)...);

				float newSlope = (1./6.) * (k1.slope + 2.*k2.slope + 2.*k3.slope + k4.slope);
				float newCurve = (1./6.) * (k1.curve + 2.*k2.curve + 2.*k3.curve + k4.curve);

				return Self{
					.val = val + newSlope * dt,
					.slope = slope + newCurve * dt,
					.curve = newCurve,
				};
			}

			T val{}; //Current value
			T slope{}; // first derivative
			T curve{}; // second derivative
		};
	};

	template<typename Real = double>
	struct Pole {
		const Real length{};
		const Real mass{};

		rk::FourthOrder<Real> horiz{};
		rk::FourthOrder<Real> angle{};
	};

	template<typename Real = double, size_t PoleCount = 1>
	struct CartPoleSystem {
		//Pole pivot offset and cart size are only for graphical purposes
		//Pole physics don't epend on their position/offset, and trackSize only needs
		// to represent the effective length of track the cart's center can travel

		rk::FourthOrder<Real> horiz;
		Pole<Real> poles[PoleCount];

		const Real trackSize;
		const Real cartMass;
		const Real gravity;
		const Real maxEngineForce;

		const Real momentDampen = 1. / 3.; // 0 <= x <= 1;

		void step(Real extForce, double dt) noexcept {
			constexpr static auto sq = [](auto x) { return x * x; };
			Real mc = cartMass;

			Real inverseMoment = 1. / (1. + momentDampen);
			Real M = mc;
			Real term1 = 0;
			Real term2 = 0;
			Real bottom1 = 0;

			for(auto const &pole: poles) {
				Real m = pole.mass;
				Real l = pole.length;
				double sin, cos;
				sincos(pole.angle.val, &sin, &cos);

				term1 += m * sin * cos;
				term2 += m * sin * (l/2.) * sq(pole.angle.slope);
				bottom1 += m * sq(cos) * pole.angle.val;

				M += m;
			}

			auto nextHorizCurve = ((gravity * term1) - inverseMoment * (extForce + term2))  /  (bottom1 - inverseMoment * M);

			this->horiz = this->horiz.step(dt, [nextHorizCurve](auto&& ...) -> Real {
				return nextHorizCurve;
			});

			for(auto &pole: poles) {
				Real l = pole.length;

				pole.angle = pole.angle.step(dt, [&](rk::FourthOrder<Real> state, auto&& ...) -> Real {
					double sin, cos;
					sincos(state.val, &sin, &cos);

					return (inverseMoment / (l/2.)) * (gravity * sin - nextHorizCurve * cos);
				});				
			}
		}
	};
}


unsigned int loadSensors(
	experiments::cart_pole_single::CartPoleSystem<double, 1> sys,
	std::vector<double> &inputs
) {
	inputs[0] = sys.horiz.val / sys.trackSize;
	inputs[1] = (sys.horiz.slope / sys.trackSize) * .1;
	inputs[2] = (sys.poles[0].angle.val / M_PI) / .1;
	inputs[3] = sys.poles[0].angle.slope / M_PI;
}

unsigned int tryout(ann::NeuralNetFast &nn, experiments::cart_pole_single::CartPoleSystem<double, 1> sys) {
	unsigned int steps = 0;
	std::vector<double> inputs(4, 0.);
	std::vector<double> outputs(1, 0.);

	for(bool stop = false; !stop && (steps < maxSteps); ++steps) {
		loadSensors(sys, inputs);
		nn.evaluate(inputs, outputs);

		double force = sys.maxEngineForce * outputs[0];
		sys.step(force, .01);

		if (sys.trackSize < fabs(sys.horiz.val)) {
			stop = true;
			break;
		}
		for(auto const &pole: sys.poles) {
			if (.1 < fabs(pole.angle.val / M_PI)) {
				stop = true;
				break;
			}
		}
	}

	return steps;
}




#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>

#include <chrono>

void show(ann::NeuralNetFast &nn, experiments::cart_pole_single::CartPoleSystem<double, 1> sys) noexcept {
	constexpr static int px_m = 30;

	sf::RenderWindow window;
	window.create(sf::VideoMode({512, 512}), "AILib", sf::Style::Default);

	sf::RectangleShape rCart(sf::Vector2f(128, 64));
	auto rCartSize = rCart.getSize();
	rCart.setOrigin(rCartSize / 2.f);
	rCart.setFillColor(sf::Color(150, 50, 250));

	sf::RectangleShape rPoles[1] = {
		sf::RectangleShape{sf::Vector2f(px_m * sys.poles[0].length, 5)}
	};
	rPoles[0].setOutlineColor(sf::Color(250, 150, 100));
	rPoles[0].rotate(sf::degrees(-90.));

	sf::Vector2u winSize = window.getSize();
	rCart.setPosition({winSize.x / 2.f, winSize.y - (rCartSize.y / 2.f) - 20.f});

	unsigned int steps = 0;
	bool stop = false;
	std::vector<double> inputs(4, 0.);
	std::vector<double> outputs(1, 0.);
	sf::Clock clock;

    while (window.isOpen()) {
    	if (stop || !(steps < maxSteps)) {
    		window.close();
    		break;
    	}

    	if (.01 < clock.getElapsedTime().asSeconds()) {
    		clock.restart();
    		++steps;

			loadSensors(sys, inputs);
			nn.evaluate(inputs, outputs);

			double force = sys.maxEngineForce * outputs[0];
			sys.step(force, .01);

			rCart.setPosition({
				(winSize.x / 2.f) + px_m * sys.horiz.val,
				winSize.y - (rCartSize.y / 2.f) - 20.f
			});
			rPoles[0].setRotation(sf::degrees(-90.) - sf::radians(sys.poles[0].angle.val));

			if (sys.trackSize < fabs(sys.horiz.val)) {
				stop = true;
			}

			for(auto const &pole: sys.poles) {
				if (.1 < fabs(pole.angle.val / M_PI)) {
					stop = true;
				}
			}    		
    	}

        for (sf::Event event; window.pollEvent(event); ) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if ((event.type == sf::Event::KeyPressed) && sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
            	window.close();
            }
        }

        window.clear(sf::Color::Black);
        window.draw(rCart);
        for (auto &rPole: rPoles) {
        	window.draw(rPole, rCart.getTransform());
        }
        window.display();
    }


}

int main() {
	experiments::cart_pole_single::CartPoleSystem<double, 1> startSys{
		.horiz = {},
		.poles = {
			{
				.length = 1.0,
				.mass = 40. - 2.,
				.angle = {
					.val = M_PI * (1./360.)
				}
			}
		},


		.trackSize = 4.8,
		.cartMass = 2.0,
		.gravity = 9.81,
		.maxEngineForce = 1000.
	};


	neat::MutationRateContainer mutation_rates = {};
	neat::SpeciatingParameterContainer speciating_parameters = {};

	speciating_parameters.population = 1000;
	// speciating_parameters.stale_species = 5;

	speciating_parameters.delta_disjoint = 1.0;
	speciating_parameters.delta_weights = 0.4;
	speciating_parameters.delta_threshold = 3.0;

	mutation_rates.connection_mutate_chance = 0.8;
	mutation_rates.perturb_chance = 0.9;
	// mutation_rates.crossover_chance = 0.95;
	// mutation_rates.disable_mutation_chance = 0.75;
	mutation_rates.node_mutation_chance = 0.03;
	mutation_rates.link_mutation_chance = 0.3;
	// mutation_rates.link_mutation_chance = 0.25;


	neat::NeatPool pool(
		2 + 2, 1, 0, false
		, mutation_rates, speciating_parameters
	);

		pool.import_fromfile("Balancer-Single.neat.pool");
		for(auto &specie: pool.species) {
			for(auto &genome: specie.genomes) {
				ann::NeuralNetFast n;

				n.from_genome(genome);
				show(n, startSys);

				return 0;
			}
		}


	for(unsigned int best_fitness = 0; best_fitness < (unsigned int)((float)maxSteps * .98); pool.new_generation()) {
		unsigned int max_fitness = 0;
		unsigned int min_fitness = maxSteps;

		for(auto &specie: pool.species) {
			for(auto &genome: specie.genomes) {
				ann::NeuralNetFast n;

				n.from_genome(genome);
				genome.fitness = tryout(n, startSys);

				min_fitness = (genome.fitness < min_fitness) ? genome.fitness : min_fitness;
				max_fitness = (max_fitness < genome.fitness) ? genome.fitness : max_fitness;

				if (best_fitness < genome.fitness) {
					best_fitness = genome.fitness;
					// std::cerr << "Saving NeuralNet with HighScore=" << genome.fitness << std::endl;
					// n.export_tofile("HighScore.neat.nn");
				}

			}
		}

		std::cerr << "Generation#" << pool.generation() << ": "
			<< "RecordHigh=" << best_fitness
			<< ", " << "Vanguard=" << max_fitness
			<< ", " << "Slacker=" << min_fitness
			<< std::endl;

		if ((pool.generation() % 100) == 0) {
			pool.export_tofile("Balancer-Single.neat.pool");
		}
	}

	pool.export_tofile("Balancer-Single.neat.pool");
}



