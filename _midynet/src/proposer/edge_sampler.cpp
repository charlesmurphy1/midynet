#include "FastMIDyNet/proposer/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

void EdgeSampler::removeEdge(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);
    double edgeWeight = round(m_edgeSampler.get_weight(orderedEdge));
    if (edgeWeight == 1)
        m_edgeSampler.erase(orderedEdge);
    else
        m_edgeSampler.set_weight(orderedEdge, edgeWeight-1);
    --m_vertexWeights[orderedEdge.first];
    --m_vertexWeights[orderedEdge.second];
}

void EdgeSampler::addEdge(const BaseGraph::Edge& edge){
    auto orderedEdge = getOrderedEdge(edge);
    if (m_edgeSampler.count(orderedEdge) == 0)
        m_edgeSampler.insert(orderedEdge, 1);
    else
        m_edgeSampler.set_weight(orderedEdge, round(m_edgeSampler.get_weight(orderedEdge))+1);

    ++m_vertexWeights[orderedEdge.first];
    ++m_vertexWeights[orderedEdge.second];
}

void EdgeSampler::insertEdge(const BaseGraph::Edge& edge, double edgeWeight=1){
    auto orderedEdge = getOrderedEdge(edge);
    m_edgeSampler.insert(edge, edgeWeight);
    m_vertexWeights[edge.first] += edgeWeight;
    m_vertexWeights[edge.second] += edgeWeight;
}

void EdgeSampler::eraseEdge(const BaseGraph::Edge& edge){
    double edgeWeight = m_edgeSampler.get_weight(edge);
    m_vertexWeights[edge.first] -= edgeWeight;
    m_vertexWeights[edge.second] -= edgeWeight;
    m_edgeSampler.erase(edge);
}

void EdgeSampler::setUp(const MultiGraph& graph){
    clear();
    m_vertexWeights = std::vector<double>(graph.getSize(), 0.);
    for (auto vertex: graph){
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
            if (vertex <= neighbor.vertexIndex){
                insertEdge({vertex, neighbor.vertexIndex}, neighbor.label);
            }
        }
    }
}

}
