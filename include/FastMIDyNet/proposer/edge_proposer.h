#ifndef FAST_MIDYNET_EDGE_PROPOSER_H
#define FAST_MIDYNET_EDGE_PROPOSER_H


#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


class EdgeProposer: public Proposer<GraphMove>{
    public:
        virtual GraphMove operator()() = 0;
        void setGraph(MultiGraph& graph) { m_graph = graph; };

    protected:
        MultiGraph m_graph;
};

}

#endif
