#ifndef FAST_MIDYNET_UNIFORM_H
#define FAST_MIDYNET_UNIFORM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"

namespace FastMIDyNet{

// class UniformMultiGraphFamily: public ErdosRenyiFamily{
// protected:
//     EdgeCountMultisetPrior m_edgeCountMultisetPrior;
// public:
//     UniformMultiGraphFamily(size_t graphSize, size_t maxEdgeCount):
//     m_edgeCountMultisetPrior(maxEdgeCount),
//     ErdosRenyiFamily(graphSize, m_edgeCountMultisetPrior){ }
// };
//
// class UniformSimpleGraphFamily: public ErdosRenyiFamily{
// protected:
//     EdgeCountBinomialPrior m_edgeCountBinomialPrior;
// public:
//     UniformSimpleGraphFamily(size_t graphSize, size_t maxEdgeCount):
//     m_edgeCountBinomialPrior(maxEdgeCount),
//     ErdosRenyiFamily(graphSize, m_edgeCountBinomialPrior){ }
// };

}// end FastMIDyNet
#endif
