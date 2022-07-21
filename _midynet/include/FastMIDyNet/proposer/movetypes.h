#ifndef FAST_MIDYNET_MOVETYPES_H
#define FAST_MIDYNET_MOVETYPES_H

#include <vector>
#include <iostream>
#include <sstream>
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


struct GraphMove{
    GraphMove(std::vector<BaseGraph::Edge> removedEdges, std::vector<BaseGraph::Edge> addedEdges):
        removedEdges(removedEdges), addedEdges(addedEdges){ }
    GraphMove(){ }
    std::vector<BaseGraph::Edge> removedEdges;
    std::vector<BaseGraph::Edge> addedEdges;

    void display() const{
        std::cout << "edges removed : { ";
        for (auto e : removedEdges){
            std::cout << "{ " << e.first << ", " << e.second << "}, ";
        }
        std::cout << "}\t edges added : { ";
        for (auto e : addedEdges){
            std::cout << "{ " << e.first << ", " << e.second << "}, ";
        }
        std::cout << "}" << std::endl;

    }
};

template <typename Label>
struct LabelMove{
    LabelMove(BaseGraph::VertexIndex vertexIndex, Label prevLabel, Label nextLabel, int addedLabels=0, Level level=0):
        vertexIndex(vertexIndex), prevLabel(prevLabel),
        nextLabel(nextLabel), addedLabels(addedLabels),
        level(level){ }
    BaseGraph::VertexIndex vertexIndex;
    Label prevLabel;
    Label nextLabel;
    int addedLabels;
    Level level;

    std::string display()const{
        std::stringstream ss;
        ss << "vertex " << vertexIndex << " at level " << level << ": " << prevLabel << " -> " << nextLabel;
        ss << " (" << addedLabels << " labels added)";
        return ss.str();
    }
};

using BlockMove = LabelMove<BlockIndex>;

// struct NestedBlockMove{
//     NestedBlockMove(std::vector<BlockMove> blockMoves, int addedLayers=0):
//         blockMoves(blockMoves), addedLayers(addedLayers){ }
//     std::vector<BlockMove> blockMoves;
//     int addedLayers;
//
//     std::vector<BlockMove>::iterator begin() { return blockMoves.begin(); }
//     std::vector<BlockMove>::iterator end() { return blockMoves.end(); }
//     const BlockMove& operator[](size_t layerIdx) const { return blockMoves[layerIdx]; }
//
//     size_t size() const { return blockMoves.size(); }
//     void display()const{
//         for(auto m : blockMoves) m.display();
//     }
// };

}

#endif
