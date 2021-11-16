#ifndef FAST_MIDYNET_TYPES
#define FAST_MIDYNET_TYPES


#include <random>
#include <vector>
#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{


typedef std::mt19937_64 RNG;

typedef int VertexState;
typedef std::vector<VertexState> State;
typedef std::vector<VertexState> VertexStateSequence;
typedef std::vector<State> StateSequence;

typedef std::vector<VertexState> VertexNeighborhoodState;
typedef std::vector<VertexNeighborhoodState> VertexNeighborhoodStateSequence;
typedef std::vector<VertexNeighborhoodState> NeighborsState;
typedef std::vector<NeighborsState> NeighborsStateSequence;

typedef BaseGraph::UndirectedMultigraph MultiGraph;

typedef size_t BlockIndex;

typedef std::vector<std::tuple<BaseGraph::VertexIndex, BlockIndex, BlockIndex>> BlockMove;
typedef std::vector<BaseGraph::Edge> EdgeMoves;


struct Move{
    double acceptation = 0.;
};

struct GraphMove: public Move{
    GraphMove(EdgeMoves edgesRemoved, EdgeMoves edgesAdded):
    edgesRemoved(edgesRemoved), edgesAdded(edgesAdded){ }
    GraphMove(){ }
    EdgeMoves edgesRemoved;
    EdgeMoves edgesAdded;
};

struct PriorMove: public Move{ };

struct BlockPriorMove: public PriorMove{
    BlockMove vertexMoved;
};

} // namespace FastMIDyNet

#endif
