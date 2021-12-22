#ifndef FAST_MIDYNET_ERDOSRENYI_H
#define FAST_MIDYNET_ERDOSRENYI_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"

namespace FastMIDyNet{

class ErdosRenyiFamily: public StochasticBlockModelFamily{
protected:
    BlockSequence m_blockSeq;
    BlockDeltaPrior m_blockDeltaPrior;
    EdgeMatrixUniformPrior m_edgeMatrixUniformPrior;
public:
    ErdosRenyiFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        m_blockSeq(graphSize, 0),
        m_blockDeltaPrior(m_blockSeq),
        m_edgeMatrixUniformPrior(edgeCountPrior, m_blockDeltaPrior),
        StochasticBlockModelFamily(){
            setBlockPrior(m_blockDeltaPrior);
            setEdgeMatrixPrior(m_edgeMatrixUniformPrior);
        }
};

}// end FastMIDyNet
#endif
