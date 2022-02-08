#include "FastMIDyNet/proposer/edge_proposer/labeled_double_edge_swap.h"


namespace FastMIDyNet{

GraphMove LabeledDoubleEdgeSwapProposer::proposeRawMove() const {
    auto labePair = m_labelSampler.sample();
    auto oldEdge1 = m_labeledEdgeSampler.at(labePair)->sample();
    auto oldEdge2 = m_labeledEdgeSampler.at(labePair)->sample();

    BaseGraph::Edge newEdge1, newEdge2;
    if (labePair.first != labePair.second)
        if (m_labelSampler.getLabelOfIdx(oldEdge1.first) == m_labelSampler.getLabelOfIdx(oldEdge2.first))
            newEdge1 = {oldEdge1.first, oldEdge2.second}, newEdge2 = {oldEdge2.first, oldEdge1.second};
        else
            newEdge2 = {oldEdge1.first, oldEdge2.first}, newEdge2 = {oldEdge2.second, oldEdge1.second};
    else if ( m_swapOrientationDistribution(rng) )
        newEdge1 = {oldEdge1.first, oldEdge2.first}, newEdge2 = {oldEdge1.second, oldEdge2.second};

    else
        newEdge1 = {oldEdge1.first, oldEdge2.second}, newEdge2 = {oldEdge1.second, oldEdge2.first};
    newEdge1 = getOrderedEdge(newEdge1), newEdge2 = getOrderedEdge(newEdge2);
    return {{oldEdge1, oldEdge2}, {newEdge1, newEdge2}};
}

void LabeledDoubleEdgeSwapProposer::setUpFromGraph(const MultiGraph& graph) {
    LabeledEdgeProposer::setUpFromGraph(graph);
    for (auto vertex : graph){
            for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
            auto rs = m_labelSampler.getLabelOfIdx({vertex, neighbor.vertexIndex});
            if (m_labeledEdgeSampler.count(rs) == 0)
                m_labeledEdgeSampler.insert({rs, new EdgeSampler()});
            if (vertex <= neighbor.vertexIndex){
                m_labeledEdgeSampler.at(rs)->insertEdge({vertex, neighbor.vertexIndex}, neighbor.label);
            }
        }
    }
}

void LabeledDoubleEdgeSwapProposer::applyGraphMove(const GraphMove& move) {
    for(auto edge : move.removedEdges){
        auto rs = m_labelSampler.getLabelOfIdx(edge);
        m_labeledEdgeSampler.at(rs)->removeEdge(edge);
    }
    for(auto edge : move.addedEdges){
        auto rs = m_labelSampler.getLabelOfIdx(edge);
        m_labeledEdgeSampler.at(rs)->addEdge(edge);
    }
}
void LabeledDoubleEdgeSwapProposer::applyBlockMove(const BlockMove& move) {
    if (move.addedBlocks == 1)
        onLabelCreation(move);

    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        auto oldPair = getOrderedPair<BlockIndex>({move.prevBlockIdx, m_labelSampler.getLabelOfIdx(neighbor.vertexIndex)});
        double weight = m_labeledEdgeSampler.at(oldPair)->eraseEdge({move.vertexIdx, neighbor.vertexIndex});
        auto newPair = getOrderedPair<BlockIndex>({move.prevBlockIdx, m_labelSampler.getLabelOfIdx(neighbor.vertexIndex)});
        m_labeledEdgeSampler.at(newPair)->insertEdge({move.vertexIdx, neighbor.vertexIndex}, weight);
    }

    if (move.addedBlocks == -1)
        onLabelDeletion(move);
}

}
