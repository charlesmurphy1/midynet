#ifndef FAST_MIDYNET_LIKELIHOOD_ERDOSRENYI_H
#define FAST_MIDYNET_LIKELIHOOD_ERDOSRENYI_H

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/prior/erdosrenyi/edge_count.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class ErdosRenyiLikelihood: public GraphLikelihoodModel{
protected:

    const double getLogLikelihoodFromEdgeCount(size_t edgeCount) const {
        size_t N = m_graphPtr->getSize();
        size_t A = (*m_withSelfLoopsPtr) ? N * (N + 1) / 2 : N * (N - 1) / 2;
        return (*m_withParallelEdgesPtr) ? logMultisetCoefficient(A, edgeCount) : logBinomialCoefficient(A, edgeCount);
    };
public:

    const double getLogLikelihood() const{
        return getLogLikelihoodFromEdgeCount((*m_edgeCountPriorPtrPtr)->getState());
    }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
        int dE = move.addedEdges.size() - move.removedEdges.size();
        size_t E = (*m_edgeCountPriorPtrPtr)->getState();
        return getLogLikelihoodFromEdgeCount(E + dE) - getLogLikelihoodFromEdgeCount(E);
    }
    EdgeCountPrior** m_edgeCountPriorPtrPtr = nullptr;
    bool* m_withSelfLoopsPtr = nullptr;
    bool* m_withParallelEdgesPtr = nullptr;
};


}

#endif
