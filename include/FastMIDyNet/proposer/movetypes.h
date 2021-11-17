#ifndef FAST_MIDYNET_MOVETYPES_H
#define FAST_MIDYNET_MOVETYPES_H

#include <vector>
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


typedef std::vector<std::tuple<BaseGraph::VertexIndex, BlockIndex, BlockIndex>> BlockMove;
typedef std::vector<BaseGraph::Edge> EdgeMove;

struct Move{
    double acceptation = 0.;
};

struct GraphMove: public Move{
    GraphMove(EdgeMove edgesRemoved, EdgeMove edgesAdded):
        edgesRemoved(edgesRemoved), edgesAdded(edgesAdded){ }
    GraphMove(){ }
    EdgeMove edgesRemoved;
    EdgeMove edgesAdded;
};

struct PriorMove: public Move{ };

struct BlockPriorMove: public PriorMove{
    BlockMove vertexMoved;
};

}

#endif
