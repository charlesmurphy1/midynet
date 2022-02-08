#include "FastMIDyNet/proposer/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

void EdgeSampler::removeEdge(const BaseGraph::Edge& edge){
    // auto orderedEdge = getOrderedEdge(edge);
    if (not contains(edge))
        throw std::logic_error("Cannot remove non-exising edge.");
    double edgeWeight = round(m_edgeSampler.get_weight(edge));
    if (edgeWeight == 1)
        m_edgeSampler.erase(edge);
    else
        m_edgeSampler.set_weight(edge, edgeWeight-1);
}

void EdgeSampler::addEdge(const BaseGraph::Edge& edge){
    // auto orderedEdge = getOrderedEdge(edge);

    if (m_edgeSampler.count(edge) == 0)
        m_edgeSampler.insert(edge, 1);
    else
        m_edgeSampler.set_weight(edge, round(m_edgeSampler.get_weight(edge))+1);
}

void EdgeSampler::insertEdge(const BaseGraph::Edge& edge, double edgeWeight=1){
    // auto orderedEdge = getOrderedEdge(edge);
    if (contains(edge))
        m_edgeSampler.set_weight(edge, edgeWeight);
    else
        m_edgeSampler.insert(edge, edgeWeight);
}

double EdgeSampler::eraseEdge(const BaseGraph::Edge& edge){
    // auto edge = getOrderedEdge(edge);
    if (not contains(edge))
        throw std::logic_error("Cannot erase non-exising edge.");
    double edgeWeight = m_edgeSampler.get_weight(edge);
    m_edgeSampler.erase(edge);
    return edgeWeight;
}

}
