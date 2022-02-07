#ifndef FAST_MIDYNET_EDGE_PROPOSER_H
#define FAST_MIDYNET_EDGE_PROPOSER_H

#include <stdexcept>
#include <unordered_set>

#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet{

class EdgeProposer: public Proposer<GraphMove>{
protected:
    const bool m_allowSelfLoops;
    const bool m_allowMultiEdges;
    const size_t m_maxIteration = 100;
    const MultiGraph* m_graphPtr = nullptr;
    bool isSelfLoop(BaseGraph::Edge edge) const { return edge.first == edge.second; }
    bool isExistingEdge(BaseGraph::Edge edge) const { return m_graphPtr->getEdgeMultiplicityIdx(edge) >= 1;}
public:
    using Proposer<GraphMove>::Proposer;
    EdgeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true):
        m_allowSelfLoops(allowSelfLoops), m_allowMultiEdges(allowMultiEdges) {}
    virtual ~EdgeProposer(){}
    GraphMove proposeMove() const override {
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
    virtual GraphMove proposeRawMove() const = 0;
    virtual void setUpFromGraph( const MultiGraph& graph ) { m_graphPtr = &graph; }
    virtual void setUp(
        const RandomGraph& randomGraph
    ) { setUpFromGraph(randomGraph.getGraph()); }
    virtual const double getLogProposalProbRatio(const GraphMove& move) const = 0;
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyBlockMove(const BlockMove& move) {};
    const bool& allowSelfLoops() const { return m_allowSelfLoops; }
    const bool& allowMultiEdges() const { return m_allowMultiEdges; }

};

}

#endif
