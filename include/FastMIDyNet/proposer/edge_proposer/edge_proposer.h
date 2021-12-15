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
    const bool& acceptIsolated() const { return m_withIsolatedVertices; }
    virtual void acceptIsolated(bool accept) { m_withIsolatedVertices = accept;}

};

}

#endif
