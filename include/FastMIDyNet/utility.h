#ifndef FAST_MIDYNET_UTILITY_H
#define FAST_MIDYNET_UTILITY_H


#include "FastMIDyNet/types.h"


size_t getDegreeIdx(const FastMIDyNet::MultiGraph& graph, size_t vertex);
std::vector<size_t> getDegrees(const FastMIDyNet::MultiGraph& graph);
double logFactorial(int);
double logDoubleFactorial(int);
double logBinomial(int);

void assertValidProbability(double probability);

#endif
