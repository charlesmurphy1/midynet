#ifndef FAST_MIDYNET_MOVETYPES_H
#define FAST_MIDYNET_MOVETYPES_H

#include <vector>
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


struct GraphMove{
    GraphMove(std::vector<BaseGraph::Edge> removedEdges, std::vector<BaseGraph::Edge> addedEdges):
        removedEdges(removedEdges), addedEdges(addedEdges){ }
    GraphMove(){ }
    std::vector<BaseGraph::Edge> removedEdges;
    std::vector<BaseGraph::Edge> addedEdges;
};

struct BlockMove{
    BlockMove(BaseGraph::VertexIndex vertexIdx, BlockIndex prevBlockIdx, BlockIndex nextBlockIdx, int addedBlocks=0):
        vertexIdx(vertexIdx), prevBlockIdx(prevBlockIdx), nextBlockIdx(nextBlockIdx), addedBlocks(addedBlocks){ }
    BaseGraph::VertexIndex vertexIdx;
    BlockIndex prevBlockIdx;
    BlockIndex nextBlockIdx;
    int addedBlocks;
};

}

#endif
