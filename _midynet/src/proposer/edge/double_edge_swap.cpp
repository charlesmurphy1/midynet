#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge/double_edge_swap.h"


namespace FastMIDyNet {


const GraphMove DoubleEdgeSwapProposer::proposeRawMove() const {
    auto edge1 = m_edgeSampler.sample();
    m_edgeSampler.onEdgeRemoval(edge1);
    auto edge2 = m_edgeSampler.sample();
    m_edgeSampler.onEdgeAddition(edge1);

    BaseGraph::Edge newEdge1, newEdge2;
    if (m_swapOrientationDistribution(rng)) {
        newEdge1 = {edge1.first, edge2.first};
        newEdge2 = {edge1.second, edge2.second};
    }
    else {
        newEdge1 = {edge1.first, edge2.second};
        newEdge2 = {edge1.second, edge2.first};
    }

    GraphMove move = {{edge1, edge2}, {newEdge1, newEdge2}};
    return move;
}

void DoubleEdgeSwapProposer::setUpWithGraph( const MultiGraph& graph ) {
    m_edgeSampler.clear();
    m_graphPtr = &graph;
    for (auto vertex : graph)
        for (auto neighbor : graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                m_edgeSampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
}

void DoubleEdgeSwapProposer::applyGraphMove(const GraphMove& move) {
    for (auto edge: move.removedEdges) {
        edge = getOrderedEdge(edge);
        m_edgeSampler.onEdgeRemoval(edge);
    }
    for (auto edge: move.addedEdges) {
        edge = getOrderedEdge(edge);
        m_edgeSampler.onEdgeAddition(edge);
    }
}


const double DoubleEdgeSwapProposer::getLogProposalProbRatio(const GraphMove& move) const{
    const auto& removedEdge1 = getOrderedEdge(move.removedEdges[0]);
    const auto& removedEdge2 = getOrderedEdge(move.removedEdges[1]);

    // move.display();
    if ( isTrivialMove(move) ){
        // printf("Trivial move\n\n");
        return 0;
    } else if ( isSelfLoop(removedEdge1) and isSelfLoop(removedEdge2) ){
        // printf("Double loopy move\n\n");
        return getLogPropForDoubleLoopyMove(move);
    } else if ( isSelfLoop(removedEdge1) or isSelfLoop(removedEdge2) ) {
        // printf("Single loopy move\n\n");
        return getLogPropForNormalMove(move) - log(2);
    } else if ( isHingeMove(move) ) {
        // printf("Hinge move\n\n");
        return getLogPropForNormalMove(move) + log(2);
    } else if ( removedEdge1 == removedEdge2 ) {
        // printf("Double edge move\n\n");
        return getLogPropForDoubleEdgeMove(move);
    }

    // printf("Normal move\n\n");
    return getLogPropForNormalMove(move);
}

bool DoubleEdgeSwapProposer::isTrivialMove(const GraphMove& move) const {
    const auto& addedEdge1 = getOrderedEdge(move.addedEdges[0]);
    const auto& addedEdge2 = getOrderedEdge(move.addedEdges[1]);

    const auto& removedEdge1 = getOrderedEdge(move.removedEdges[0]);
    const auto& removedEdge2 = getOrderedEdge(move.removedEdges[1]);

    if (addedEdge1 == removedEdge1 and addedEdge2 == removedEdge2)
        return true;
    if (addedEdge1 == removedEdge2 and addedEdge2 == removedEdge1)
        return true;
    return false;
}

bool DoubleEdgeSwapProposer::isHingeMove(const GraphMove& move) const {
    const auto& removedEdge1 = getOrderedEdge(move.removedEdges[0]);
    const auto& removedEdge2 = getOrderedEdge(move.removedEdges[1]);
    const auto& i = removedEdge1.first, j = removedEdge1.second;
    const auto& k = removedEdge2.first, l = removedEdge2.second;
    return (i == k and j != l) or
           (j == k and i != l) or
           (i == l and j != k) or
           (j == l and i != k);
}


const double DoubleEdgeSwapProposer::getLogPropForNormalMove(const GraphMove& move) const{
    const auto& addedEdge1 = getOrderedEdge(move.addedEdges[0]), addedEdge2 = getOrderedEdge(move.addedEdges[1]);
    const auto& removedEdge1 = getOrderedEdge(move.removedEdges[0]), removedEdge2 = getOrderedEdge(move.removedEdges[1]);
    double addedEdge1Weight = m_edgeSampler.getEdgeWeight(addedEdge1);
    double addedEdge2Weight = m_edgeSampler.getEdgeWeight(addedEdge2);
    double removedEdge1Weight = m_edgeSampler.getEdgeWeight(removedEdge1);
    double removedEdge2Weight = m_edgeSampler.getEdgeWeight(removedEdge2);
    return log(addedEdge1Weight + 1) + log(addedEdge2Weight + 1)
         - log(removedEdge1Weight) - log(removedEdge2Weight);
}

const double DoubleEdgeSwapProposer::getLogPropForDoubleLoopyMove(const GraphMove& move) const{
    const auto& addedEdge = getOrderedEdge(move.addedEdges[0]);
    const auto& removedSelfLoop1 = getOrderedEdge(move.removedEdges[0]);
    const auto& removedSelfLoop2 = getOrderedEdge(move.removedEdges[1]);
    double addedEdgeWeight = m_edgeSampler.getEdgeWeight(addedEdge);
    double removedSelfLoop1Weight = m_edgeSampler.getEdgeWeight(removedSelfLoop1);
    double removedSelfLoop2Weight = m_edgeSampler.getEdgeWeight(removedSelfLoop2);
    return log(addedEdgeWeight + 2) + log(addedEdgeWeight + 1)
         - log(removedSelfLoop1Weight) - log(removedSelfLoop2Weight)
         - log(4);
}

const double DoubleEdgeSwapProposer::getLogPropForDoubleEdgeMove(const GraphMove& move) const{
    const auto& addedSelfLoop1 = getOrderedEdge(move.addedEdges[0]);
    const auto& addedSelfLoop2 = getOrderedEdge(move.addedEdges[1]);
    const auto& removedEdge = getOrderedEdge(move.removedEdges[0]);
    double addedSelfLoop1Weight = m_edgeSampler.getEdgeWeight(addedSelfLoop1);
    double addedSelfLoop2Weight = m_edgeSampler.getEdgeWeight(addedSelfLoop2);
    double removedEdgeWeight = m_edgeSampler.getEdgeWeight(removedEdge);
    return log(4) + log(addedSelfLoop1Weight + 1) + log(addedSelfLoop2Weight + 1)
         - log(removedEdgeWeight) - log(removedEdgeWeight - 1);
}



} // namespace FastMIDyNet
