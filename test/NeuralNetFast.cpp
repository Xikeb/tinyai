#include <gtest/gtest.h>
#include <gmock/gmock.h>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}


#include <stdint.h> // For int32_t, etc.

union FloatingPointType
{
    FloatingPointType(float num = double(0.0)) : f(num) {}
    // Portable extraction of components.
    bool Negative() const { return i < 0; }
    int32_t RawMantissa() const { return i & ((1 << 23) - 1); }
    int32_t RawExponent() const { return (i >> 23) & 0xFF; }

    bool AlmostEquals(FloatingPointType const &oth) const noexcept {
      if (this->Negative() != oth.Negative()) {
        return this->f == oth.f;
      }

      int ulpsDiff = abs(this->i - oth.i);

      return ulpsDiff <= 4;
    }

    int32_t i;
    float f;
#ifdef _DEBUG
    struct
    {   // Bitfields for exploration. Do not use in production code.
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } parts;
#endif
};

union DoubleType
{
    DoubleType(double num = double(0.0)) : f(num) {}
    // Portable extraction of components.
    bool Negative() const { return i < 0; }
    int64_t RawMantissa() const { return i & ((1 << 52) - 1); }
    int64_t RawExponent() const { return (i >> 52) & 0xFF; }

    bool AlmostEquals(DoubleType const &oth) const noexcept {
      if (this->Negative() != oth.Negative()) {
        return this->f == oth.f;
      }

      int ulpsDiff = abs(this->i - oth.i);

      return ulpsDiff <= 4;
    }

    int64_t i;
    double f;
#ifdef _DEBUG
    struct
    {   // Bitfields for exploration. Do not use in production code.
        uint64_t mantissa : 52;
        uint64_t exponent : 11;
        uint64_t sign : 1;
    } parts;
#endif
};

bool AlmostEquals(float a, float b) noexcept {
  return FloatingPointType(a).AlmostEquals(FloatingPointType(b));
}

bool AlmostEquals(double a, double b) noexcept {
  return DoubleType(a).AlmostEquals(DoubleType(b));
}


#include "Phenotypes/IPhenotype.hpp"
#include "NeuralNetFast.hpp"
#include "Genotype.hpp"

using ::testing::AllOf;
using ::testing::Ge;
using ::testing::Le;
using ::testing::MatchesRegex;
using ::testing::StartsWith;

struct NeuralNetActivationTest: public testing::TestWithParam<double> {
public:

  NeuralNetActivationTest(): info{}, mutationRates{}, genotype(info, mutationRates) {
  }

private:
  neat::NetworkInfoContainer info;
  neat::MutationRateContainer mutationRates;
  neat::Genotype genotype;
  ann::NeuralNetFast nn;
};

// Demonstrate some basic assertions.
TEST(NeuralNetActivation, PhenotypeSigmoidActivation) {
  for(double y = -20.; y <= 20.; y += 0.1) {
    ASSERT_THAT(
      ann::sigmoid(y),
      AllOf(Ge(-1.), Le(1.))
    ) << "Activation function must be constrained to -1 <= f(a1 + ... + an) <= 1";
  }
}

// INSTANTIATE_TEST_SUITE_P(ActivationIsConstrainedToAbsolute1,
//                          NeuralNetActivationTest,
//                          testing::Range(-20., 20., 0.1));
