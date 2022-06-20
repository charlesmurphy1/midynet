#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

const GraphMove EdgeProposer::proposeMove() const {
    for (size_t i = 0; i < m_maxIteration; i++) {
        GraphMove move = proposeRawMove();
        for (auto e : move.addedEdges){
            if ((isSelfLoop(e) and not m_allowSelfLoops) or (isExistingEdge(e) and not m_allowMultiEdges))
                continue;
            return move;
        }
    }
    throw std::runtime_error("EdgeProposer: Could not find edge to propose.");
}

}
