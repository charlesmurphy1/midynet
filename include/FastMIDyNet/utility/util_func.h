#ifndef FAST_MIDYNET_UTIL_FUNC_H
#define FAST_MIDYNET_UTIL_FUNC_H

#include "FastMIDyNet/types.h"

namespace FastMIDyNet {

double logFactorial(int);
double logDoubleFactorial(int);
double logBinomial(int);
double logPoissonPMF(size_t x, double mean);
double logRestrictedPartitionNumber(size_t n, size_t k);
double logRestrictedPartitionNumber(size_t n, size_t k);
double logApproxRestrictedPartitionNumber(size_t n, size_t k);

} // namespace FastMIDyNet

#endif
