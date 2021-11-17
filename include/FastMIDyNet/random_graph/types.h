#ifndef FAST_MIDYNET_RANDOM_GRAPH_TYPES_H
#define FAST_MIDYNET_RANDOM_GRAPH_TYPES_H

#include <vector>
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{

typedef size_t BlockIndex;
typedef std::vector<int> DegreeSequence;
typedef std::vector<BlockIndex> BlockSequence;
typedef Matrix<size_t> EdgeMatrix;

}

#endif
