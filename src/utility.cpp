#include <stdexcept>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility.h"


using namespace std;


namespace FastMIDyNet {

size_t getDegreeIdx(const FastMIDyNet::MultiGraph& graph, size_t vertex) {
    size_t degree = 0;

    for (auto neighbor: graph.getNeighboursOfIdx(vertex))
        if (neighbor.first == vertex)
            degree += 2*neighbor.second;
        else
            degree += neighbor.second;
    return degree;
}

DegreeSequence getDegrees(const FastMIDyNet::MultiGraph& graph) {
    DegreeSequence degrees(graph.getSize());
    for (size_t vertex=0; vertex<graph.getSize(); vertex++)
        degrees[vertex] = getDegreeIdx(graph, vertex);
    return degrees;
}

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

double logBinomial(size_t n, size_t k){
    if (n >= k) throw invalid_argument("`n` must be greater or equal to `k`.");
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}

double logMultinom(vector<size_t> k){
    double result = 0;
    size_t sum = 0;
    for (auto kk : k){
        result -= logFactorial(kk);
        sum += kk;
    }
    result += logFactorial(sum);
    return result;
}

void assertValidProbability(double probability) {
    if (probability > 1 || probability < 0)
        throw std::invalid_argument("Invalid probability "+std::to_string(probability)+
                ". Probability must be contained between 0 and 1.");
}

double logPoissonPMF(size_t x, double mean) {
    return x*log(mean) - logFactorial(x) - mean;
}

} // namespace FastMIDyNet
