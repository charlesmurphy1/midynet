#include <iostream>
#include <math.h>
#include <list>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"


using namespace std;

namespace FastMIDyNet {



double logFactorial(size_t n){
    return lgamma(n + 1);
}

double logDoubleFactorial(size_t n){
    size_t k;
    if ( n%2 == 0 ){
        k = n / 2;
        return k * log(2) + logFactorial(k);
    }else{
        k = (n + 1) / 2;
        return logFactorial(2 * k) - k * log(2) - logFactorial(k);
    }
}

double logBinomialCoefficient(size_t n, size_t k){
    if (n < k) throw invalid_argument("`n` must be greater or equal to `k`: "
    + to_string(n) + " !> " + to_string(k));
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}


double logMultinomialCoefficient(std::list<size_t> sequence) {
    size_t sumSequence=0;
    size_t sumLGammaSequencePlusOne=0;
    for (size_t element: sequence) {
        sumSequence += element;
        sumLGammaSequencePlusOne += lgamma(element + 1);
    }
    return lgamma(sumSequence + 1) - sumLGammaSequencePlusOne;
}
double logMultisetCoefficient(size_t n, size_t k){
    return logBinomialCoefficient(n + k - 1, k);
}

double logPoissonPMF(size_t x, double mean) {
    return x*log(mean) - logFactorial(x) - mean;
}

double logZeroTruncatedPoissonPMF(size_t x, double mean) {
    return x*log(mean) - logFactorial(x) - mean - log(1 - exp(-mean));
}

BaseGraph::Edge getOrderedEdge(const BaseGraph::Edge& edge) {
    if (edge.first < edge.second)
        return edge;
    return {edge.second, edge.first};
}

void assertValidProbability(double probability) {
    if (probability > 1 || probability < 0)
        throw ConsistencyError("Probability " + std::to_string(probability) + " is not between 0 and 1.");
}

std::pair<size_t, size_t> getUndirectedPairFromIndex(size_t index, size_t n) {
    // const size_t i = floor(-.5 + sqrt(.25+2*index));
    // const size_t j = index - i*(i-1)*.5;
    const size_t i = index / n;
    const size_t j = index % n;
    return {i, j};
}

} // namespace FastMIDyNet
