#ifndef FAST_MIDYNET_EDGE_PROPOSER_UTIL_H
#define FAST_MIDYNET_EDGE_PROPOSER_UTIL_H


#include <map>
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"


namespace FastMIDyNet{

using LabelPair = std::pair<BlockIndex, BlockIndex>;
std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> getSubGraphOfLabelPair(const RandomGraph& randomGraph);

void checkEdgeSamplerConsistencyWithGraph(const std::string, const MultiGraph&, const EdgeSampler&);

void checkVertexSamplerConsistencyWithGraph(const std::string, const MultiGraph&, const VertexSampler&);

}

#endif
