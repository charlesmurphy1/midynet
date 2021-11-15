#ifndef FAST_MIDYNET_GENERATORS
#define FAST_MIDYNET_GENERATORS

#include <random>
#include <vector>
#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


int generateCategorical(std::vector<double> probs, RNG rng);

} // namespace FastMIDyNet

#endif
