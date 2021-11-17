#ifndef FAST_MIDYNET_UTILITY_H
#define FAST_MIDYNET_UTILITY_H


#include <random>
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


extern RNG rng;

size_t getDegreeIdx(const FastMIDyNet::MultiGraph& graph, size_t vertex);
DegreeSequence getDegrees(const FastMIDyNet::MultiGraph& graph);
double logFactorial(int);
double logDoubleFactorial(int);
double logBinomial(int);

void assertValidProbability(double probability);

}


#endif
