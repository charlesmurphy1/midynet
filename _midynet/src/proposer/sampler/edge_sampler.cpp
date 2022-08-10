#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

void EdgeSampler::setUpWithGraph(const MultiGraph& graph) {
    clear();
    m_graphPtr = &graph;
    for (const auto& vertex : graph){
        for (const auto& neighbor : graph.getNeighboursOfIdx(vertex)){
            if (vertex <= neighbor.vertexIndex)
                onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
        }
    }
}

void EdgeSampler::onEdgeRemoval(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);
    if (not contains(orderedEdge))
        throw std::runtime_error("EdgeSampler: Cannot remove non-exising edge ("
            + std::to_string(orderedEdge.first) + ", "
            + std::to_string(orderedEdge.second) + ").");

    double weight = getEdgeWeight(edge);
    if (weight == 1)
        m_edgeSampler.erase(orderedEdge);
    else if (m_graphPtr->getEdgeMultiplicityIdx(orderedEdge) <= m_maxWeight)
        m_edgeSampler.set_weight(orderedEdge, weight-1);
}

void EdgeSampler::onEdgeAddition(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);

    double weight = getEdgeWeight(orderedEdge);
    if (not contains(orderedEdge))
        m_edgeSampler.insert(orderedEdge, 1);
    else if (m_graphPtr->getEdgeMultiplicityIdx(orderedEdge) < m_maxWeight and weight < m_maxWeight)
        m_edgeSampler.set_weight(orderedEdge, weight+1);
}

void EdgeSampler::onEdgeInsertion(const BaseGraph::Edge& edge, double weight=1){
    auto orderedEdge = getOrderedEdge(edge);
    weight = (weight >= m_maxWeight) ? m_maxWeight : weight;
    if (contains(orderedEdge))
        m_edgeSampler.set_weight(orderedEdge, weight);
    else
        m_edgeSampler.insert(orderedEdge, weight);
}

double EdgeSampler::onEdgeErasure(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);
    if (not contains(orderedEdge))
        throw std::runtime_error("EdgeSampler: Cannot erase non-exising edge ("
            + std::to_string(orderedEdge.first) + ", "
            + std::to_string(orderedEdge.second) + ").");
    double weight = m_edgeSampler.get_weight(orderedEdge);
    m_edgeSampler.erase(orderedEdge);
    return weight;
}

}
