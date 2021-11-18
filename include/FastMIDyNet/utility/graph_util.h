#ifndef FAST_MIDYNET_UTIL_GRAPH_H
#define FAST_MIDYNET_UTIL_GRAPH_H

#include "FastMIDyNet/types.h"

namespace FastMIDyNet {

size_t getDegreeIdx(const FastMIDyNet::MultiGraph&, size_t vertex);
DegreeSequence getDegrees(const FastMIDyNet::MultiGraph&);

} // namespace FastMIDyNet

#endif
