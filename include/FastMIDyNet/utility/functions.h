#ifndef FAST_MIDYNET_UTIL_FUNCTIONS_H
#define FAST_MIDYNET_UTIL_FUNCTIONS_H

#include <list>
#include <utility>
#include "FastMIDyNet/types.h"

namespace FastMIDyNet {

// static const double INFINITY = std::numeric_limits<double>::infinity();

double logFactorial(size_t);
double logDoubleFactorial(size_t);
double logBinomialCoefficient(size_t, size_t);
double logPoissonPMF(size_t x, double mean);
double logMultinomialCoefficient(std::list<size_t> sequence);
double logMultisetCoefficient(size_t n, size_t k);

double logRestrictedPartitionNumber(size_t n, size_t k);
double logRestrictedPartitionNumber(size_t n, size_t k);
double logApproxRestrictedPartitionNumber(size_t n, size_t k);

template<typename T>
std::pair<T, T> getOrderedPair(const std::pair<T, T>& myPair){
    if (myPair.first < myPair.second)
        return myPair;
    return {myPair.second, myPair.first};
}

BaseGraph::Edge getOrderedEdge(const BaseGraph::Edge& edge) ;

} // namespace FastMIDyNet

#endif
