#ifndef FAST_MIDYNET_TYPES
#define FAST_MIDYNET_TYPES


#include <random>
#include <vector>
#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{


typedef std::mt19937_64 RNG;

template<typename T>
using Matrix=std::vector<std::vector<T>>;

typedef BaseGraph::UndirectedMultigraph MultiGraph;

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

} // namespace FastMIDyNet

#endif
