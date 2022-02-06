#include "FastMIDyNet/proposer/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

void EdgeSampler::update(const GraphMove& move){
    for (auto edge: move.removedEdges) {
        edge = getOrderedEdge(edge);
        size_t edgeWeight = round(m_edgeSampler.get_weight(edge));
        if (edgeWeight == 1)
            m_edgeSampler.erase(edge);
        else
            m_edgeSampler.set_weight(edge, edgeWeight-1);
        --m_degrees[edge.first];
        --m_degrees[edge.second];
    }

    for (auto edge: move.addedEdges) {
        edge = getOrderedEdge(edge);
        if (m_edgeSampler.count(edge) == 0)
            m_edgeSampler.insert(edge, 1);
        else
            m_edgeSampler.set_weight(edge, round(m_edgeSampler.get_weight(edge))+1);

        ++m_degrees[edge.first];
        ++m_degrees[edge.second];
    }
}

void EdgeSampler::setUp(const MultiGraph& graph,
    const std::unordered_set<BaseGraph::VertexIndex>& vertexBlackList,
    const std::unordered_set<BaseGraph::Edge>& edgeWhiteList){
    m_edgeSampler.clear();
    m_degrees = graph.getDegrees();
    for (auto vertex: graph) {
        if (vertexBlackList.count(vertex) > 0)
            continue;

        for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
            BaseGraph::Edge edge = {vertex, neighbor.vertexIndex};
            if (vertex <= neighbor.vertexIndex and
                vertexBlackList.count(neighbor.vertexIndex) == 0 and
                (edgeWhiteList.size() == 0 or edgeWhiteList.count(edge) > 0)
             )
                m_edgeSampler.insert(edge, neighbor.label);
        }
    }
}

}
