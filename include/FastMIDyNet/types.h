#ifndef FAST_MIDYNET_TYPES_H
#define FAST_MIDYNET_TYPES_H


#include <map>
#include <random>
#include <vector>

#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{


template<typename T>
using Matrix=std::vector<std::vector<T>>;


typedef std::mt19937_64 RNG;


typedef BaseGraph::UndirectedMultigraph MultiGraph;
typedef size_t BlockIndex;
typedef std::vector<size_t> DegreeSequence;
typedef std::vector<BlockIndex> BlockSequence;
typedef Matrix<size_t> EdgeMatrix;
typedef Matrix<size_t> EdgeMatrix;

} // namespace FastMIDyNet

#endif
