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
    mutable std::uniform_real_distribution<double> m_uniform01;
    bool isSelfLoop(BaseGraph::Edge edge) const { return edge.first == edge.second; }
    bool isExistingEdge(BaseGraph::Edge edge) const { return m_graphPtr->getEdgeMultiplicityIdx(edge) >= 1;}
public:
    using Proposer<GraphMove>::Proposer;
    EdgeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true):
        m_allowSelfLoops(allowSelfLoops), m_allowMultiEdges(allowMultiEdges) {}
    virtual ~EdgeProposer(){}
    GraphMove proposeMove() const override ;
    virtual GraphMove proposeRawMove() const = 0;
    virtual const double getLogProposalProbRatio(const GraphMove& move) const = 0;
    const GraphMove getReverseMove(const GraphMove& move) const {
        return {move.addedEdges, move.removedEdges};
    }
    virtual void setUp( const RandomGraph& randomGraph ) { clear(); setUpFromGraph(randomGraph.getGraph()); }
    virtual void setUpFromGraph( const MultiGraph& graph ) { m_graphPtr = &graph; }
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyBlockMove(const BlockMove& move) {};
    const bool& allowSelfLoops() const { return m_allowSelfLoops; }
    const bool& allowMultiEdges() const { return m_allowMultiEdges; }

    virtual void clear() override { m_graphPtr = nullptr; }

};

}

#endif
