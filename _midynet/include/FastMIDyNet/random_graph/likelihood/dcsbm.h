#ifndef FAST_MIDYNET_LIKELIHOOD_DCSBM_H
#define FAST_MIDYNET_LIKELIHOOD_DCSBM_H

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/random_graph/prior/labeled_degree.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class DegreeCorrectedStochasticBlockModelLikelihood: public VertexLabeledGraphLikelihoodModel<BlockIndex>{
protected:

    void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;
    void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>>&) const;
    void getDiffEdgeMatMapFromBlockMove(const BlockMove&, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;
    const double getLogLikelihoodRatioEdgeTerm (const GraphMove&) const;
    const double getLogLikelihoodRatioAdjTerm (const GraphMove&) const;

public:
    const double getLogLikelihood() const override;
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove&) const override;
    const double getLogLikelihoodRatioFromLabelMove (const BlockMove&) const override;
    VertexLabeledDegreePrior** m_degreePriorPtrPtr = nullptr;

};

}

#endif
