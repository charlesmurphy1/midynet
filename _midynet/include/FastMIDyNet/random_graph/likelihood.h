#ifndef FAST_MIDYNET_LIKELIHOOD_H
#define FAST_MIDYNET_LIKELIHOOD_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class StochasticBlockModelLikelihood{
protected:
    size_t* m_sizePtr = nullptr;
    MultiGraph* m_graphPtr = nullptr;
    BlockPrior** m_blockPriorPtrPtr = nullptr;
    EdgeMatrixPrior** m_edgeMatrixPriorPtrPtr = nullptr;

    void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;
    void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>>&) const;
    void getDiffEdgeMatMapFromBlockMove(const BlockMove&, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;


    virtual const double getLogLikelihood() const ;
    virtual const double getLogLikelihoodRatioEdgeTerm (const GraphMove&) const;
    virtual const double getLogLikelihoodRatioAdjTerm (const GraphMove&) const;
    virtual const double getLogLikelihoodRatioFromGraphMove (const GraphMove&) const ;
    virtual const double getLogLikelihoodRatioFromLabelMove (const BlockMove&) const ;
};


class DegreeCorrectedStochasticBlockModelLikelihood: public StochasticBlockModelLikelihood{
protected:
    DegreePrior** m_degreePriorPtrPtr = nullptr;

    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioEdgeTerm (const GraphMove&) const override;
    const double getLogLikelihoodRatioAdjTerm (const GraphMove&) const override;
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove&) const override;
    const double getLogLikelihoodRatioFromLabelMove (const BlockMove&) const override;

};

}

#endif
