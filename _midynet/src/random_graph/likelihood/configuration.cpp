#include "FastMIDyNet/random_graph/likelihood/configuration.h"

namespace FastMIDyNet{

const double ConfigurationModelLikelihood::getLogLikelihood() const{
    const size_t& E = (*m_degreePriorPtrPtr)->getEdgeCount();
    const auto& degrees =  (*m_degreePriorPtrPtr)->getState();
    double logLikelihood = logDoubleFactorial(2 * E) - logFactorial(2 * E);

    for (auto vertex : *m_statePtr){
        logLikelihood += logFactorial(degrees[vertex]);
        for (auto neighbor : m_statePtr->getNeighboursOfIdx(vertex)){
            if (vertex > neighbor.vertexIndex)
                continue;
            logLikelihood -= (neighbor.vertexIndex == vertex) ? logDoubleFactorial(2 * neighbor.label) : logFactorial(neighbor.label);
        }
    }

    return logLikelihood;
}

const double ConfigurationModelLikelihood::getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
    IntMap<size_t> degreeDiffMap;
    IntMap<BaseGraph::Edge> edgeMultDiffMap;
    const auto& degrees = (*m_degreePriorPtrPtr)->getState();
    int dE = move.addedEdges.size() - move.removedEdges.size();
    const size_t& E = (*m_degreePriorPtrPtr)->getEdgeCount();



    for (auto edge : move.addedEdges){
        edgeMultDiffMap.increment(getOrderedEdge(edge));
        degreeDiffMap.increment(edge.first);
        degreeDiffMap.increment(edge.second);
    }

    for (auto edge : move.removedEdges){
        edgeMultDiffMap.decrement(getOrderedEdge(edge));
        degreeDiffMap.decrement(edge.first);
        degreeDiffMap.decrement(edge.second);
    }

    double logLikelihoodRatio = logDoubleFactorial(2 * (E + dE)) - logFactorial(2 * (E + dE));
    logLikelihoodRatio -= logDoubleFactorial(2 * E) - logFactorial(2 * E);
    for (auto diff : degreeDiffMap){
        logLikelihoodRatio += logFactorial(degrees[diff.first] + diff.second) - logFactorial(degrees[diff.first]);
    }

    for (auto diff : edgeMultDiffMap){
        size_t edgeMult = m_statePtr->getEdgeMultiplicityIdx(diff.first);
        int factor = (diff.first.first == diff.first.second) ? 2 : 1;
        auto factFunc = (diff.first.first == diff.first.second) ? logDoubleFactorial : logFactorial;
        logLikelihoodRatio -= factFunc(factor * (edgeMult + diff.second)) - factFunc(factor * edgeMult);
    }

    return logLikelihoodRatio;
}

}
