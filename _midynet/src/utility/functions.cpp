#include <iostream>
#include <math.h>
#include <list>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/polylog2_integral.h"
#include "FastMIDyNet/exceptions.h"


using namespace std;

namespace FastMIDyNet {

const size_t MAX_INTEGER_THRESHOLD = 500;

double logFactorial(size_t n){
    if (n < MAX_INTEGER_THRESHOLD) return lgamma(n + 1);
    else return 0.5 * sqrt(2 * PI * n) + n * (log(n) - 1);
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
    if (n < k) throw invalid_argument("logBinomialCoefficient: `n` (" + to_string(n) +
    ") must be greater or equal to `k` (" + to_string(k) + ").");
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}


double logMultinomialCoefficient(std::list<size_t> sequence) {
    size_t sumSequence = 0;
    double sumLFactorialSequence = 0;
    for (size_t element: sequence) {
        sumSequence += element;
        sumLFactorialSequence += logFactorial(element);
    }
    return logFactorial(sumSequence) - sumLFactorialSequence;
}

double logMultinomialCoefficient(std::vector<size_t> sequence) {
    size_t sumSequence = 0;
    double sumLFactorialSequence = 0;
    for (size_t element: sequence) {
        sumSequence += element;
        sumLFactorialSequence += logFactorial(element);
    }
    return logFactorial(sumSequence) - sumLFactorialSequence;
}

double logMultisetCoefficient(size_t n, size_t k){
    if (n == 0){
        return 0;
    } else {
        return logBinomialCoefficient(n + k - 1, k);
    }
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

std::list<BaseGraph::Edge> getEdgeList(const MultiGraph& graph){
    std::list<BaseGraph::Edge> edgeList;
    for (auto vertex : graph)
        for (auto neighbor : graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                for (size_t l=0; l < neighbor.label; ++l)
                    edgeList.push_back({vertex, neighbor.vertexIndex});
    return edgeList;
}

std::map<BaseGraph::Edge, size_t> getWeightedEdgeList(const MultiGraph& graph){
    std::map<BaseGraph::Edge, size_t> edgeList;
    for (auto vertex : graph)
        for (auto neighbor : graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                edgeList.insert({{vertex, neighbor.vertexIndex}, neighbor.label});
    return edgeList;
}

void assertValidProbability(double probability) {
    if (probability > 1 || probability < 0)
        throw ConsistencyError("Probability " + std::to_string(probability) + " is not between 0 and 1.");
}

std::pair<size_t, size_t> getUndirectedPairFromIndex(size_t index, size_t n) {
    const size_t i = floor(-.5 + sqrt(.25+2*index));
    const size_t j = index - i*(i+1)*.5;
    return {j, i};
}

MultiGraph getSubGraphByBlocks(const MultiGraph& graph, const BlockSequence& blocks, BlockIndex r, BlockIndex s){
    MultiGraph subGraph(graph.getSize());

    for (auto vertex : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(vertex)){
            if ((vertex < neighbor.vertexIndex) && (blocks[vertex] == r && blocks[neighbor.vertexIndex] == s))
                subGraph.setEdgeMultiplicityIdx(vertex, neighbor.vertexIndex, neighbor.label);
        }
    }

    return subGraph;
}

double clip(double x, double min, double max){
    if (x < min)
        return min;
    else if (x > max)
        return max;
    else
        return x;
}

double clipProb(double p, double epsilon){ return clip(p, epsilon, 1 - epsilon); }

void displayNeighborhood(const MultiGraph&graph, const BaseGraph::VertexIndex& v){
    std::cout << "vertex " << v << ": ";
    for ( auto neighbor : graph.getNeighboursOfIdx(v))
        std::cout << "(" << neighbor.vertexIndex << ", " << neighbor.label << ") ";
    std::cout << std::endl;
}
void displayGraph(const MultiGraph&graph, std::string name){
    std::cout << name << ":" << std::endl;
    for ( auto v : graph){
        std::cout << "\t";
        displayNeighborhood(graph, v);
    }
}

} // namespace FastMIDyNet
