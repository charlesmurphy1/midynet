#ifndef FAST_MIDYNET_LIKELIHOOD_ERDOSRENYI_H
#define FAST_MIDYNET_LIKELIHOOD_ERDOSRENYI_H

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class ErdosRenyiLikelihood: public GraphLikelihoodModel{
protected:

    const double getLogLikelihoodFromEdgeCount(size_t edgeCount) const {
        size_t N = *m_graphSizePtr;
        size_t A = (*m_withSelfLoopsPtr) ? N * (N + 1) / 2 : N * (N - 1) / 2;
        return (*m_withParallelEdgesPtr) ? logMultisetCoefficient(A, edgeCount) : logBinomialCoefficient(A, edgeCount);
    };
public:

    const MultiGraph sample() const {
        const auto& generate = (*m_withParallelEdgesPtr) ? generateMultiGraphErdosRenyi: generateErdosRenyi;
        return generate(*m_graphSizePtr, (*m_edgeCountPriorPtrPtr)->getState(), *m_withSelfLoopsPtr);
    }
    const double getLogLikelihood() const{
        return getLogLikelihoodFromEdgeCount((*m_edgeCountPriorPtrPtr)->getState());
    }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
        int dE = move.addedEdges.size() - move.removedEdges.size();
        size_t E = (*m_edgeCountPriorPtrPtr)->getState();
        return getLogLikelihoodFromEdgeCount(E + dE) - getLogLikelihoodFromEdgeCount(E);
    }
    EdgeCountPrior** m_edgeCountPriorPtrPtr = nullptr;
    size_t* m_graphSizePtr = nullptr;
    bool* m_withSelfLoopsPtr = nullptr;
    bool* m_withParallelEdgesPtr = nullptr;
};


}

#endif
