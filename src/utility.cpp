#include "FastMIDyNet/utility.h"


size_t getDegreeIdx(const FastMIDyNet::MultiGraph& graph, size_t vertex) {
    size_t degree = 0;

    for (auto neighbor: graph.getNeighboursOfIdx(vertex))
        if (neighbor.first == vertex)
            degree += 2*neighbor.second;
        else
            degree += neighbor.second;
    return degree;
}

std::vector<size_t> getDegrees(const FastMIDyNet::MultiGraph& graph) {
    std::vector<size_t> degrees(graph.getSize());
    for (size_t vertex=0; vertex<graph.getSize(); vertex++)
        degrees[vertex] = getDegreeIdx(graph, vertex);
    return degrees;
}

void assertValidProbability(double probability) {
    if (probability > 1 || probability < 0)
        throw std::invalid_argument("Invalid probability "+std::to_string(probability)+
                ". Probability must be contained between 0 and 1.");
}
