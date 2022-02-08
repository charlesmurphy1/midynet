#include "FastMIDyNet/proposer/edge_proposer/labeled_hinge_flip.h"


namespace FastMIDyNet{

GraphMove LabeledHingeFlipProposer::proposeRawMove() const {
    auto labelPair = m_labelSampler.sample();
    auto edge = m_labeledEdgeSampler.at(labelPair)->sample();
    BlockIndex r;
    BaseGraph::VertexIndex commonVertex, losingVertex, gainingVertex;
    if ( m_swapOrientationDistribution(rng) ){
        r = m_labelSampler.getLabelOfIdx(edge.first);
        commonVertex = edge.first;
        losingVertex = edge.second;
    }
    else{
        r = m_labelSampler.getLabelOfIdx(edge.second);
        commonVertex = edge.second;
        losingVertex = edge.first;
    }
    gainingVertex = m_labeledVertexSampler.at(r)->sample();

    return {{{commonVertex, losingVertex}}, {{commonVertex, gainingVertex}}};

}

void LabeledHingeFlipProposer::setUpFromGraph(const MultiGraph& graph) {
    LabeledEdgeProposer::setUpFromGraph(graph);
    for (auto vertex : graph){
         auto r = m_labelSampler.getLabelOfIdx(vertex);
         if (m_labeledVertexSampler.count(r) == 0)
            m_labeledVertexSampler.insert({r, constructVertexSampler()});
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
            auto rs = m_labelSampler.getLabelOfIdx({vertex, neighbor.vertexIndex});
            if (m_labeledEdgeSampler.count(rs) == 0)
                m_labeledEdgeSampler.insert({rs, new EdgeSampler()});
            if (vertex <= neighbor.vertexIndex){
                m_labeledVertexSampler.at(r)->insertEdge({vertex, -1}, neighbor.label);
                m_labeledEdgeSampler.at(rs)->insertEdge({vertex, neighbor.vertexIndex}, neighbor.label);
            }
        }
    }
}

void LabeledHingeFlipProposer::applyGraphMove(const GraphMove& move) {
    for(auto edge : move.removedEdges){
        auto rs = m_labelSampler.getLabelOfIdx(edge);
        m_labeledEdgeSampler.at(rs)->removeEdge(edge);
        m_labeledVertexSampler.at(rs.first)->removeEdge(edge);
        m_labeledVertexSampler.at(rs.second)->removeEdge(edge);
    }
    for(auto edge : move.addedEdges){
        auto rs = m_labelSampler.getLabelOfIdx(edge);
        m_labeledEdgeSampler.at(rs)->addEdge(edge);
        m_labeledVertexSampler.at(rs.first)->addEdge(edge);
        m_labeledVertexSampler.at(rs.second)->addEdge(edge);
    }
}
void LabeledHingeFlipProposer::applyBlockMove(const BlockMove& move) {
    if (move.addedBlocks == 1)
        onLabelCreation(move);

    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){

    }

    if (move.addedBlocks == -1)
        onLabelDeletion(move);
}

}
