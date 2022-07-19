#ifndef FAST_MIDYNET_RANDOMGRAPH_UTIL_H
#define FAST_MIDYNET_RANDOMGRAPH_UTIL_H

#include <string>
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

MultiGraph getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq);
void checkGraphConsistencyWithEdgeMatrix( std::string namePrefix, const MultiGraph& graph, const BlockSequence& blockSeq, const MultiGraph& expectedEdgeMat);
void checkGraphConsistencyWithDegreeSequence(std::string namePrefix, const MultiGraph& graph, const DegreeSequence& expectedDegreeSeq);

}

#endif
