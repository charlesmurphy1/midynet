#ifndef FAST_MIDYNET_UTILITY_H
#define FAST_MIDYNET_UTILITY_H


#include <random>
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


extern RNG rng;

size_t getDegreeIdx(const FastMIDyNet::MultiGraph&, size_t vertex);
DegreeSequence getDegrees(const FastMIDyNet::MultiGraph&);
double logFactorial(int);
double logDoubleFactorial(int);
double logBinomial(int);

double logPoissonPMF(size_t x, double mean);

void assertValidProbability(double probability);

}


#endif
