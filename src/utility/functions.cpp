
#include <math.h>
#include <list>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"


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
    if (n >= k) throw invalid_argument("`n` must be greater or equal to `k`.");
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

double logPoissonPMF(size_t x, double mean) {
    return x*log(mean) - logFactorial(x) - mean;
}

BaseGraph::Edge getOrderedEdge(const BaseGraph::Edge& edge) {
    if (edge.first < edge.second)
        return edge;
    return {edge.second, edge.first};
}

} // namespace FastMIDyNet
