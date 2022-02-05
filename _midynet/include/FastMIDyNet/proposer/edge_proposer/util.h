#ifndef FAST_MIDYNET_EDGE_PROPOSER_UTIL_H
#define FAST_MIDYNET_EDGE_PROPOSER_UTIL_H


#include <map>
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet{

using LabelPair = std::pair<BlockIndex, BlockIndex>;
std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> getSubGraphOfLabelPair(const RandomGraph& randomGraph);

}

#endif
