#ifndef FAST_MIDYNET_MOVETYPES_H
#define FAST_MIDYNET_MOVETYPES_H

#include <vector>
#include <iostream>
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


struct GraphMove{
    GraphMove(std::vector<BaseGraph::Edge> removedEdges, std::vector<BaseGraph::Edge> addedEdges):
        removedEdges(removedEdges), addedEdges(addedEdges){ }
    GraphMove(){ }
    std::vector<BaseGraph::Edge> removedEdges;
    std::vector<BaseGraph::Edge> addedEdges;

    void display(){

        std::cout << "edges added : { ";
        for (auto e : addedEdges){
            std::cout << "{ " << e.first << ", " << e.second << "}, ";
        }
        std::cout << "}\t edges removed : { ";

        for (auto e : removedEdges){
            std::cout << "{ " << e.first << ", " << e.second << "}, ";
        }
        std::cout << "}" << std::endl;

    }
};

struct BlockMove{
    BlockMove(BaseGraph::VertexIndex vertexIdx, BlockIndex prevBlockIdx, BlockIndex nextBlockIdx, int addedBlocks=0):
        vertexIdx(vertexIdx), prevBlockIdx(prevBlockIdx), nextBlockIdx(nextBlockIdx), addedBlocks(addedBlocks){ }
    BaseGraph::VertexIndex vertexIdx;
    BlockIndex prevBlockIdx;
    BlockIndex nextBlockIdx;
    int addedBlocks;

    void display(){
        std::cout << "vertex " << vertexIdx << ": " << prevBlockIdx << " -> " << nextBlockIdx << std::endl;
    }
};

}

#endif
