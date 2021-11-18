#include <stdexcept>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/graph_util.h"


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

}
