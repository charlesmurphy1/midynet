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
        m_labeledVertexSampler.at(r)->onVertexInsertion(vertex);
    }
    for (auto vertex : graph){
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
            auto rs = m_labelSampler.getLabelOfIdx({vertex, neighbor.vertexIndex});
            if (m_labeledEdgeSampler.count(rs) == 0)
                m_labeledEdgeSampler.insert({rs, new EdgeSampler()});
            if (vertex <= neighbor.vertexIndex){
                BaseGraph::Edge edge = {vertex, neighbor.vertexIndex};
                m_labeledVertexSampler.at(rs.first)->onEdgeInsertion(edge, neighbor.label);
                m_labeledVertexSampler.at(rs.second)->onEdgeInsertion(edge, neighbor.label);
                m_labeledEdgeSampler.at(rs)->onEdgeInsertion(edge, neighbor.label);
            }
        }
    }
}


void LabeledHingeFlipProposer::applyGraphMove(const GraphMove& move) {
    for(auto edge : move.removedEdges){
        edge = getOrderedEdge(edge);
        auto rs = m_labelSampler.getLabelOfIdx(edge);
        m_labeledVertexSampler.at(rs.first)->onEdgeRemoval(edge);
        m_labeledVertexSampler.at(rs.second)->onEdgeRemoval(edge);
        m_labeledEdgeSampler.at(rs)->onEdgeRemoval(edge);
    }
    for(auto edge : move.addedEdges){
        edge = getOrderedEdge(edge);
        auto rs = m_labelSampler.getLabelOfIdx(edge);
        m_labeledVertexSampler.at(rs.first)->onEdgeAddition(edge);
        m_labeledVertexSampler.at(rs.second)->onEdgeAddition(edge);
        m_labeledEdgeSampler.at(rs)->onEdgeAddition(edge);
    }
}
void LabeledHingeFlipProposer::applyBlockMove(const BlockMove& move) {
    if (move.addedBlocks == 1)
        onLabelCreation(move);

    m_labeledVertexSampler.at(move.nextBlockIdx)->onVertexInsertion( move.vertexIdx);
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        BaseGraph::Edge edge = getOrderedEdge({move.vertexIdx, neighbor.vertexIndex});
        auto s = m_labelSampler.getLabelOfIdx(neighbor.vertexIndex);
        LabelPair prevLabelPair = getOrderedPair<BlockIndex>({move.prevBlockIdx, s});
        LabelPair nextLabelPair = getOrderedPair<BlockIndex>({move.nextBlockIdx, s});
        if (move.prevBlockIdx != s)
            m_labeledVertexSampler.at(move.prevBlockIdx)->onEdgeErasure(edge);
        if (move.nextBlockIdx != s)
            m_labeledVertexSampler.at(move.nextBlockIdx)->onEdgeInsertion(edge, neighbor.label);
        m_labeledEdgeSampler.at(prevLabelPair)->onEdgeErasure(edge);
        if (m_labeledEdgeSampler.count(nextLabelPair) == 0)
            m_labeledEdgeSampler.insert({nextLabelPair, new EdgeSampler()});
        m_labeledEdgeSampler.at(nextLabelPair)->onEdgeInsertion(edge, neighbor.label);
    }
    m_labeledVertexSampler.at(move.prevBlockIdx)->onVertexErasure(move.vertexIdx);


    if (move.addedBlocks == -1)
        onLabelDeletion(move);
}

void LabeledHingeFlipProposer::clear(){
    LabeledEdgeProposer::clear();
    for (auto p : m_labeledEdgeSampler)
        delete p.second;
    m_labeledEdgeSampler.clear();

    for (auto p : m_labeledVertexSampler)
        delete p.second;
    m_labeledVertexSampler.clear();
}

const double LabeledHingeFlipDegreeProposer::getLogProposalProbRatio(const GraphMove& move) const {
    auto losingVertex = move.removedEdges[0].second, gainingVertex = move.addedEdges[0].second ;
    auto r = m_labelSampler.getLabelOfIdx(losingVertex);
    return log(m_labeledVertexSampler.at(r)->getVertexWeight(losingVertex) - 1) -
           log(m_labeledVertexSampler.at(r)->getVertexWeight(gainingVertex));
}


}
