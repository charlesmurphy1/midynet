#ifndef FAST_MIDYNET_EDGE_PROPOSER_H
#define FAST_MIDYNET_EDGE_PROPOSER_H


#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{

class EdgeProposer: public Proposer<GraphMove>{
public:
    virtual void setUp(const RandomGraph& randomGraph) { };

};

}

#endif
