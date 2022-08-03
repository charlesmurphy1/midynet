#ifndef FAST_MIDYNET_RANDOMGRAPH_UTIL_H
#define FAST_MIDYNET_RANDOMGRAPH_UTIL_H

#include <string>
#include "FastMIDyNet/types.h"
#include "prior/edge_count.h"
#include "prior/block_count.h"
#include "prior/block.h"
#include "prior/degree.h"
#include "prior/labeled_degree.h"
#include "likelihood/sbm.h"

namespace FastMIDyNet{

MultiGraph getLabelGraphFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq);
void checkGraphConsistencyWithLabelGraph( std::string namePrefix, const MultiGraph& graph, const BlockSequence& blockSeq, const MultiGraph& expectedEdgeMat);
void checkGraphConsistencyWithDegreeSequence(std::string namePrefix, const MultiGraph& graph, const DegreeSequence& expectedDegreeSeq);
EdgeCountPrior* makeEdgeCountPrior(double edgeCount, bool canonical=false);
BlockPrior* makeBlockPrior(size_t size, BlockCountPrior& blockCountPrior, bool hyperPrior=false);
DegreePrior* makeDegreePrior(size_t size, EdgeCountPrior& prior, bool hyperPrior=false);
VertexLabeledDegreePrior* makeVertexLabeledDegreePrior(LabelGraphPrior& prior, bool hyperPrior=false);

StochasticBlockModelLikelihood* makeSBMLikelihood(bool stubLabeled=true);

}

#endif
