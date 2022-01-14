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
        DegreeCorrectedStochasticBlockModelFamily(graphSize),
        m_blockSeq(graphSize, 0),
        m_blockDeltaPrior(m_blockSeq),
        m_edgeMatrixUniformPrior() {
            setBlockPrior(m_blockDeltaPrior);
            setEdgeMatrixPrior(m_edgeMatrixUniformPrior);
        }
    ConfigurationModelFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior, DegreePrior& degreePrior):
        DegreeCorrectedStochasticBlockModelFamily(graphSize),
        m_blockSeq(graphSize, 0),
        m_blockDeltaPrior(m_blockSeq),
        m_edgeMatrixUniformPrior(edgeCountPrior, m_blockDeltaPrior){
            setBlockPrior(m_blockDeltaPrior);
            setEdgeMatrixPrior(m_edgeMatrixUniformPrior);
            setDegreePrior(degreePrior);
        }
    void setEdgeCountPrior(EdgeCountPrior& edgeCountPrior){
        m_edgeMatrixUniformPrior.setEdgeCountPrior(edgeCountPrior);
    }
    void setDegreePrior(DegreePrior& degreePrior) {
        m_degreePriorPtr = &degreePrior;
        m_degreePriorPtr->isRoot(false);
        m_degreePriorPtr->setBlockPrior(m_blockDeltaPrior);
        m_degreePriorPtr->setEdgeMatrixPrior(m_edgeMatrixUniformPrior);
    }
};

}// end FastMIDyNet
#endif
