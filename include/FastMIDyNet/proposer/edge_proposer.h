#ifndef FAST_MIDYNET_EDGE_PROPOSER_H
#define FAST_MIDYNET_EDGE_PROPOSER_H


#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


class EdgeProposer: Proposer<GraphMove>{
    public:
        const GraphMove& operator()();
};

}

#endif
