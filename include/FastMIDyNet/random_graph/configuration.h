#ifndef FAST_MIDYNET_CONFIGURATION_H
#define FAST_MIDYNET_CONFIGURATION_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/dcsbm.h"

namespace FastMIDyNet{

class ConfigurationModelFamily: public DegreeCorrectedStochasticBlockModelFamily{
protected:
    EdgeMatrixUniformPrior m_edgeMatrixUniformPrior;
    BlockSequence m_blockSeq;
    BlockDeltaPrior m_blockDeltaPrior;
public:
    ConfigurationModelFamily(DegreePrior& degreePrior):
        m_blockSeq(degreePrior.getSize(), 0),
        m_blockDeltaPrior(m_blockSeq),
        m_edgeMatrixUniformPrior(degreePrior.getEdgeMatrixPrior().getEdgeCountPrior(), m_blockDeltaPrior),
        DegreeCorrectedStochasticBlockModelFamily(m_blockDeltaPrior, m_edgeMatrixUniformPrior, degreePrior){ }
};

}// end FastMIDyNet
#endif
