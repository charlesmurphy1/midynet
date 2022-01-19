#include "FastMIDyNet/proposer/edge_proposer/vertex_sampler.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"


namespace FastMIDyNet{

void VertexUniformSampler::setUp(const MultiGraph& graph){
    for (auto vertex : graph){
        if (m_withIsolatedVertices or graph.getDegreeOfIdx(vertex) > 0)
            m_vertexSampler.insert(vertex, 1);
    }
}


BaseGraph::VertexIndex VertexDegreeSampler::sample() const {
    auto edge = m_edgeSampler.sample_ext_RNG(rng).first;

    if ( m_vertexChoiceDistribution(rng) )
        return edge.first;
    else
        return edge.second;
}

void VertexDegreeSampler::setUp(const MultiGraph& graph){
    m_edgeSampler.clear();
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                m_edgeSampler.insert({vertex, neighbor.vertexIndex}, neighbor.label);
    m_degrees = graph.getDegrees();
}

void VertexDegreeSampler::update(const GraphMove& move) {
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

}
