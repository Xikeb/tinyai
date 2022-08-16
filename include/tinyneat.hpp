#ifndef __TINYNEAT_HPP__
#define __TINYNEAT_HPP__

/* custom defines:
 * INCLUDE_ENABLED_GENES_IF_POSSIBLE  - if during experiment you found that too many genes are
 *                                      disabled, you can use this option.
 * ALLOW_RECURRENCY_IN_NETWORK	      - allowing recurrent links 
 *
 * GIVING_NAMES_FOR_SPECIES           - giving species unique names (need a dictionary with 
 *                                      names in a file "specie_names.dict"
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>
#include <list>
#include <string>

#include "MutationRateContainer.hpp"
#include "SpeciatingParameterContainer.hpp"
#include "NetworkInfoContainer.hpp"
#include "Gene.hpp"
#include "Genotype.hpp"
#include "Specie.hpp"
#include "InnovationContainer.hpp"
#include "NeatPool.hpp"

#endif
