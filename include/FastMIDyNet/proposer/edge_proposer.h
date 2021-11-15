#ifndef FAST_MIDYNET_EDGE_PROPOSER_H
#define FAST_MIDYNET_EDGE_PROPOSER_H


#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


class EdgeProposer: public Proposer<GraphMove>{
    public:
        GraphMove operator()();
        void setGraph(MultiGraph& graph) { m_graph = graph; };

    protected:
        MultiGraph m_graph;
};

}

#endif
