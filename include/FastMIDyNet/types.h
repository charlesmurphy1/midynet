#ifndef FAST_MIDYNET_TYPES
#define FAST_MIDYNET_TYPES

#include <random>
#include <vector>
#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{

typedef std::mt19937_64 RNGType;
typedef std::vector<int> StateType;
typedef std:::vector<std::vector<int>> NeighborsStateType;
typedef BaseGraph::UndirectedMultigraph GraphType;
typedef std::vector<Edge> EdgeMoveType;

struct GraphMoveType{
    EdgeMoveType edges_removed;
    EdgeMoveType edges_added;
    double acceptation = 0.;
}

} // namespace FastMIDyNet

#endif
