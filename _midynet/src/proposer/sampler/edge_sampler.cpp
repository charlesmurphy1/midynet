#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

void EdgeSampler::onEdgeRemoval(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);
    if (not contains(orderedEdge))
        throw std::logic_error("EdgeSampler: Cannot remove non-exising edge ("
            + std::to_string(orderedEdge.first) + ", "
            + std::to_string(orderedEdge.second) + ").");
    double edgeWeight = round(m_edgeSampler.get_weight(orderedEdge));
    if (edgeWeight == 1)
        m_edgeSampler.erase(orderedEdge);
    else
        m_edgeSampler.set_weight(orderedEdge, edgeWeight-1);
}

void EdgeSampler::onEdgeAddition(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);

    if (m_edgeSampler.count(orderedEdge) == 0)
        m_edgeSampler.insert(orderedEdge, 1);
    else
        m_edgeSampler.set_weight(orderedEdge, round(m_edgeSampler.get_weight(orderedEdge))+1);
}

void EdgeSampler::onEdgeInsertion(const BaseGraph::Edge& edge, double edgeWeight=1){
    auto orderedEdge = getOrderedEdge(edge);
    if (contains(orderedEdge))
        m_edgeSampler.set_weight(orderedEdge, edgeWeight);
    else
        m_edgeSampler.insert(orderedEdge, edgeWeight);
}

double EdgeSampler::onEdgeErasure(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);
    if (not contains(orderedEdge))
        throw std::logic_error("EdgeSampler: Cannot erase non-exising edge ("
            + std::to_string(orderedEdge.first) + ", "
            + std::to_string(orderedEdge.second) + ").");
    double edgeWeight = m_edgeSampler.get_weight(orderedEdge);
    m_edgeSampler.erase(orderedEdge);
    return edgeWeight;
}

}
