#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"

namespace FastMIDyNet {

void RandomGraph::_applyGraphMove(const GraphMove& move){
    for (auto edge: move.addedEdges){
        auto v = edge.first, u = edge.second;
        m_state.addEdgeIdx(v, u);
    }
    for (auto edge: move.removedEdges){
        auto v = edge.first, u = edge.second;
        if ( m_state.isEdgeIdx(u, v) )
            m_state.removeEdgeIdx(v, u);
        else
            throw std::runtime_error("Cannot remove non-existing edge (" + std::to_string(u) + ", " + std::to_string(v) + ").");
    }

}


void RandomGraph::setUp() {
    m_edgeProposerPtr->setUpWithPrior(*this);
}

const double RandomGraph::getLogProposalRatioFromGraphMove (const GraphMove& move) const{
    return m_edgeProposerPtr->getLogProposalProbRatio(move);
}


void RandomGraph::applyGraphMove(const GraphMove& move) {
    processRecursiveFunction([&](){ _applyGraphMove(move); });
    m_edgeProposerPtr->applyGraphMove(move);
    #if DEBUG
    checkConsistency();
    #endif
}

const GraphMove RandomGraph::proposeGraphMove() const {
    return m_edgeProposerPtr->proposeMove();
}

}
