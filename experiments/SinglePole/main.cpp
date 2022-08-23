#include <vector>
#include <array>
#include <cmath>
#include <iostream>

#include "NeatPool.hpp"
#include "NeuralNet.hpp"
#include "NeuralNetFast.hpp"

namespace experiments::cart_pole_single {

	template<typename Type>
	struct coord {
		using type = Type;
		using self = coord<type>;

		type x = 0;
		type y = 0;

    	template<typename NewType>
		constexpr coord<NewType> as() const noexcept {
			return {NewType(x), NewType(y)};
		}

		constexpr type dot(coord const &b) const noexcept {
			return this->x*b.x + this->y*b.y;
		}
		constexpr type sqlen() const noexcept {
			return this->x * this->x + this->y * this->y;
		}
		type len() const noexcept {
			return std::sqrt(sqlen());
		}
		self unit() const noexcept {
			return *this / this->len();
		}
		self &normalize() const noexcept {
			return *this /= this->len();
		}

		self &round() noexcept {
			x = std::round(x);
			y = std::round(y);

			return *this;
		}

		constexpr self rounded() const noexcept {
			return {std::round(x), std::round(y)};
		}

		constexpr self project(self const &b) const noexcept {
			return *this * (this->dot(b) / b.sqlen());
		}

		constexpr coord projectNorm(coord const &b) const noexcept {
			return *this * this->norm().dot(b.norm());
		}
		coord rotate(double angle) const noexcept {
			double sin, cos;
			sincos(angle, &sin, &cos);
			return {float(x*cos -y*sin), float(x*sin+y*cos)};
		}

		constexpr bool isInRect(self const &corner1, self const &corner2) const noexcept {
			return ((corner1.x <= x) && (x < corner2.x))
			&& ((corner1.y <= y) && (y < corner2.y));
		}

		constexpr bool isInRect(type corner1_x, type corner1_y, type corner2_x, type corner2_y) const noexcept {
			return ((corner1_x <= x) && (x < corner2_x))
			&& ((corner1_y <= y) && (y < corner2_y));
		}

		constexpr bool inRange(self const &oth, double range) const noexcept {
			return (*this - oth).sqlen() <= (range * range);
		}

		constexpr bool inRange(double range) const noexcept {
			return this->sqlen() <= (range * range);
		}

		constexpr self operator-() const noexcept {
			return {type{-x}, type{-y}};
		}

    	template<typename BType>
		inline constexpr operator coord<BType>() noexcept {
			return this->as<BType>();
		}
	};

	template<typename Type>
	std::ostream &operator<<(std::ostream &os, coord<Type> const &c) {
		return os << '{' << c.x << " ; " << c.y << '}';
	}


	template<typename Type, typename B>
	inline constexpr auto operator*(coord<Type> const &c, B b) noexcept {
		using ResType = decltype(c.x * b);
		return coord<ResType>{ResType(c.x*b), ResType(c.y*b)};
	}
	template<typename Type, typename B>
	inline constexpr coord<Type> &operator*=(coord<Type> &c, B b) noexcept {
		c.x *= b;
		c.y *= b;
		return c;
	}
	template<typename Type, typename B>
	inline constexpr auto operator/(coord<Type> const &c, B b) noexcept {
		return c * (1/b);
	}
	template<typename Type, typename B>
	inline constexpr coord<Type> &operator/=(coord<Type> &c, B b) noexcept {
		return c *= (1/b);
	}

	template<typename AType, typename BType>
	inline constexpr auto operator+(coord<AType> const &a,coord<BType> const &b) noexcept {
		using Ctype = decltype(a.x + b.x);
		return coord<Ctype>{Ctype(a.x + b.x), Ctype(a.y + b.y)};
	}
	template<typename AType, typename BType>
	inline constexpr auto operator-(coord<AType> const &a,coord<BType> const &b) noexcept {
		using Ctype = decltype(a.x - b.x);
		return coord<Ctype>{Ctype(a.x - b.x), Ctype(a.y-b.y)};
	}
	template<typename AType, typename BType>
	inline constexpr coord<AType> &operator+=(coord<AType> &a,coord<BType> const &b) noexcept {
		a.x += b.x;
		a.y += b.y;
		return a;
	}
	template<typename AType, typename BType>
	inline constexpr coord<AType> &operator-=(coord<AType> &a,coord<BType> const &b) noexcept {
		a.x -= b.x;
		a.y -= b.y;
		return a;
	}

	template<typename T>
	constexpr std::pair<coord<T>, int> intersectSegments(coord<T> const &a1, coord<T> const &a2, coord<T> const &b1, coord<T> const &b2) noexcept {
		auto fa1 = a1.template as<double>(),
		fa2 = a2.template as<double>(),
		fb1 = b1.template as<double>(),
		fb2 = b2.template as<double>();

		double d = (fa1.x - fa2.x) * (fb1.y - fb2.y) - (fa1.y - fa2.y) * (fb1.x - fb2.x);
		if (d == 0) {
			return {{T(),T()}, false};
		}

		double xi = ((fb1.x - fb2.x) * (fa1.x * fa2.y - fa1.y * fa2.x) - (fa1.x - fa2.x) * (fb1.x * fb2.y - fb1.y * fb2.x)) / d;
		double yi = ((fb1.y - fb2.y) * (fa1.x * fa2.y - fa1.y * fa2.x) - (fa1.y - fa2.y) * (fb1.x * fb2.y - fb1.y * fb2.x)) / d;

		if (not ((xi < fa1.x) ^ (xi < fa2.x))) { return {{T(),T()}, false}; }
		if (not ((xi < fb1.x) ^ (xi < fb2.x))) { return {{T(),T()}, false}; }

		return {coord<double>{xi, yi}.template as<T>(), true};
	}

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
				using State = FourthOrder<T>;

				Self k1 = this->sample(std::forward<GetAcceleration>(reader),     0, std::forward<Args>(args)...);
				Self k2 =    k1.sample(std::forward<GetAcceleration>(reader), dt/2., std::forward<Args>(args)...);
				Self k3 =    k2.sample(std::forward<GetAcceleration>(reader), dt/2., std::forward<Args>(args)...);
				Self k4 =    k3.sample(std::forward<GetAcceleration>(reader),    dt, std::forward<Args>(args)...);

				float newSlope = (1./6.) * (k1.slope + 2.*k2.slope + 2.*k3.slope + k4.slope);
				float newCurve = (1./6.) * (k1.curve + 2.*k2.curve + 2.*k3.curve + k4.curve);

				return Self{
					.val = val + newSlope * dt,
					.slope = slope + newCurve * dt,
					.curve = curve,
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

		const Real momentDampen = 1; // 0 <= x <= 1;

		void step(Real force, double dt) noexcept {
			constexpr static auto sq = [](auto x) { return x * x; };
			Real mc = cartMass;
			Real cartAccel = force / mc;

			this->horiz = this->horiz.step(dt, [cartAccel](auto&& ...) -> Real {
				return cartAccel;
			});

			for(size_t i = 0; i < PoleCount; ++i) {
				auto &pole = poles[i];
				Real m = pole.mass;
				Real l = pole.length;
				Real sysMass = mc + m;
				Real ml = m * l;

				pole.angle = pole.angle.step(dt, [&](rk::FourthOrder<Real> state, auto&& ...) -> Real {
					double sin, cos;
					sincos(state.val, &sin, &cos);

					auto top = (sysMass * ml * sin * gravity)
					 		   - (ml * cos) * (force + ml * sin * sq(state.slope));
					auto bottom = sysMass * (ml * l + momentDampen) - sq(ml * cos);

					return top / bottom;
				});				
			}
		}
	};
}

unsigned int tryout(ann::NeuralNetFast &nn, experiments::cart_pole_single::CartPoleSystem<double, 1> sys) {
	unsigned int steps = 0;
	std::vector<double> inputs(4, 0.);
	std::vector<double> outputs(1, 0.);

	for(bool stop = false; !stop && (steps < 10000); ++steps) {
		inputs[0] = sys.horiz.val / sys.trackSize;
		inputs[1] = (sys.horiz.slope / sys.trackSize) * .1;
		inputs[2] = ((sys.poles[0].angle.val / M_PI) - .5) / .1;
		inputs[2] = sys.poles[0].angle.slope / M_PI;

		nn.evaluate(inputs, outputs);

		double force = sys.maxEngineForce * outputs[0];
		sys.step(force, .01);

		if (sys.trackSize < fabs(sys.horiz.val)) {
			stop = true;
			break;
		}
		for(auto const &pole: sys.poles) {
			if (.1 < fabs((pole.angle.val / M_PI) - .5)) {
				stop = true;
				break;
			}
		}
	}

	return steps;
}


int main() {
	experiments::cart_pole_single::CartPoleSystem<double, 1> startSys{
		.horiz = {},
		.poles = {
			{
				.length = 1.0,
				.mass = .1,
				.angle = {
					.val = M_PI * (.5 + (1./360.))
				}
			}
		},


		.trackSize = 4.8,
		.cartMass = 1.0,
		.gravity = 9.81,
		.maxEngineForce = 10.
	};


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
		2 + 2, 1, 0, false
		, mutation_rates, speciating_parameters
	);

	for(unsigned int best_fitness = 0; best_fitness < 990; pool.new_generation()) {
		unsigned int max_fitness = 0;
		unsigned int min_fitness = 100000;

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
	}
}