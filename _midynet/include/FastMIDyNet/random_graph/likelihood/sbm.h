#ifndef FAST_MIDYNET_LIKELIHOOD_SBM_H
#define FAST_MIDYNET_LIKELIHOOD_SBM_H

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/random_graph/prior/edge_matrix.h"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class StochasticBlockModelLikelihood: public VertexLabeledGraphLikelihoodModel<BlockIndex>{
protected:

    void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;
    void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>>&) const;
    void getDiffEdgeMatMapFromBlockMove(const BlockMove&, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;
    const double getLogLikelihoodRatioEdgeTerm (const GraphMove&) const;
    const double getLogLikelihoodRatioAdjTerm (const GraphMove&) const;

public:
    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove&) const ;
    const double getLogLikelihoodRatioFromLabelMove (const BlockMove&) const ;
    EdgeMatrixPrior** m_edgeMatrixPriorPtrPtr = nullptr;
};


}

#endif
