#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/distance.h"


namespace FastMIDyNet{

double HammingDistance::compute(const MultiGraph& graph1, const MultiGraph& graph2) const{
    double distance = 0;

    for (auto vertexIdx : graph1)
        for (auto neighbor: graph1.getNeighboursOfIdx(vertexIdx))
            if (graph2.getEdgeMultiplicityIdx(vertexIdx, neighbor.vertexIndex) == 0)
                ++distance;
    for (auto vertexIdx : graph2)
        for (auto neighbor: graph2.getNeighboursOfIdx(vertexIdx))
            if (graph1.getEdgeMultiplicityIdx(vertexIdx, neighbor.vertexIndex) == 0)
                ++distance;
    return distance;
}

}
