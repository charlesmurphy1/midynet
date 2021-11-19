#ifndef FAST_MIDYNET_UTIL_FUNCTIONS_H
#define FAST_MIDYNET_UTIL_FUNCTIONS_H

#include <list>
#include "FastMIDyNet/types.h"

namespace FastMIDyNet {

double logFactorial(size_t);
double logDoubleFactorial(size_t);
double logBinomialCoefficient(size_t, size_t);
double logPoissonPMF(size_t x, double mean);
double logMultinomialCoefficient(std::list<size_t> sequence);
double logRestrictedPartitionNumber(size_t n, size_t k);
double logRestrictedPartitionNumber(size_t n, size_t k);
double logApproxRestrictedPartitionNumber(size_t n, size_t k);

} // namespace FastMIDyNet

#endif
