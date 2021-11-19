#ifndef FAST_MIDYNET_TYPES_H
#define FAST_MIDYNET_TYPES_H


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
typedef std::vector<int> DegreeSequence;
typedef std::vector<BlockIndex> BlockSequence;
typedef Matrix<size_t> EdgeMatrix;

struct Edge: public std::pair<size_t, size_t>  {
    Edge() { this->first=0; this->second=0; }
    Edge(const std::pair<size_t, size_t>& source) {
        if (source.first < source.second) {
            this->first = source.first;
            this->second = source.second;
        }
        else {
            this->first = source.second;
            this->second = source.first;
        }
    }
};

} // namespace FastMIDyNet

#endif
