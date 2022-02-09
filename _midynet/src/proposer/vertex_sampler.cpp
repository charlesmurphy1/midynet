#include "FastMIDyNet/proposer/vertex_sampler.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

BaseGraph::VertexIndex VertexDegreeSampler::sample() const {
    double prob = m_shift * m_vertexSampler.total_weight() / (
        m_shift * m_vertexSampler.total_weight() + m_totalEdgeWeight
    );
    if (m_uniform01(rng) < prob)
        return m_vertexSampler.sample_ext_RNG(rng).first;

    auto edge = m_edgeSampler.sample();
    if ( m_vertexChoiceDistribution(rng) or not contains(edge.second))
        return edge.first;
    else if ( contains(edge.second) )
        return edge.second;
    else
        throw std::logic_error("Proposed edge (" + std::to_string(edge.first) + ", " + std::to_string(edge.second) + ") is invalid.");
}



void VertexDegreeSampler::onVertexInsertion(const BaseGraph::VertexIndex& vertex) {
    if (not contains(vertex)){
        m_vertexSampler.insert(vertex, 1);
        m_weights.insert({vertex, 0});
    }
}

void VertexDegreeSampler::onVertexErasure(const BaseGraph::VertexIndex& vertex) {
    if (not contains(vertex))
        throw std::logic_error("Cannot remove non-exising vertex " + std::to_string(vertex) + ".");
    m_vertexSampler.erase(vertex);
    m_weights.erase(vertex);
}

void VertexDegreeSampler::onEdgeInsertion(const BaseGraph::Edge& edge, double edgeWeight) {
    m_edgeSampler.onEdgeInsertion(edge, edgeWeight);
    m_totalEdgeWeight += edgeWeight;
    if ( contains(edge.first) )
        m_weights[edge.first] += edgeWeight;
    if ( contains(edge.second) )
        m_weights[edge.second] += edgeWeight;

}

void VertexDegreeSampler::onEdgeErasure(const BaseGraph::Edge& edge) {
    if (not contains(edge.first) and not contains(edge.second))
        return;
    double edgeWeight = m_edgeSampler.onEdgeErasure(edge);
    m_totalEdgeWeight -= edgeWeight;

    if ( contains(edge.first) )
        m_weights[edge.first] -= edgeWeight;
    if ( contains(edge.second) )
        m_weights[edge.second] -= edgeWeight;
}

void VertexDegreeSampler::onEdgeAddition(const BaseGraph::Edge& edge) {
    m_edgeSampler.onEdgeAddition(edge);
    ++m_totalEdgeWeight;
    if ( contains(edge.first) )
        ++m_weights[edge.first];
    if ( contains(edge.second) )
        ++m_weights[edge.second];
}

void VertexDegreeSampler::onEdgeRemoval(const BaseGraph::Edge& edge) {
    if (not contains(edge.first) and not contains(edge.second))
        return;
    m_edgeSampler.onEdgeRemoval(edge);
    --m_totalEdgeWeight;
    if ( contains(edge.first) )
        --m_weights[edge.first];
    if ( contains(edge.second) )
        --m_weights[edge.second];
}

}
