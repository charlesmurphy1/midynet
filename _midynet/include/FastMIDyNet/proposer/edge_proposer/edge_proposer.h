#ifndef FAST_MIDYNET_EDGE_PROPOSER_H
#define FAST_MIDYNET_EDGE_PROPOSER_H


#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet{

class EdgeProposer: public Proposer<GraphMove>{
protected:
    bool m_withIsolatedVertices = true;
public:
    virtual void setUp(const RandomGraph& randomGraph) = 0;
    bool getAcceptIsolated() const { return m_withIsolatedVertices; }
    virtual bool setAcceptIsolated(bool accept) { return m_withIsolatedVertices = accept; }
    virtual double getLogProposalProbRatio(const GraphMove& move) const = 0;
    virtual void updateProbabilities(const GraphMove& move) {};
    virtual void updateProbabilities(const BlockMove& move) {};

};

}

#endif
