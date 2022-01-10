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
    BlockSequence m_blockSeq;
    BlockDeltaPrior m_blockDeltaPrior;
    EdgeMatrixUniformPrior m_edgeMatrixUniformPrior;
public:
    ConfigurationModelFamily(size_t graphSize):
        DegreeCorrectedStochasticBlockModelFamily(graphSize) {}
    ConfigurationModelFamily(size_t graphSize, DegreePrior& degreePrior):
        DegreeCorrectedStochasticBlockModelFamily(graphSize),
        m_blockSeq(graphSize, 0),
        m_blockDeltaPrior(m_blockSeq),
        m_edgeMatrixUniformPrior(degreePrior.getEdgeMatrixPrior().getEdgeCountPriorRef(), m_blockDeltaPrior){
            setBlockPrior(m_blockDeltaPrior);
            setEdgeMatrixPrior(m_edgeMatrixUniformPrior);
            setDegreePrior(degreePrior);
        }
};

}// end FastMIDyNet
#endif