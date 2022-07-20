#include "FastMIDyNet/random_graph/likelihood/dcsbm.h"
#include "BaseGraph/types.h"

using namespace BaseGraph;
namespace FastMIDyNet{
void DegreeCorrectedStochasticBlockModelLikelihood::getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge& edge, int counter, IntMap<std::pair<BlockIndex, BlockIndex>>& diffEdgeMatMap) const{
    const BlockSequence& blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
    diffEdgeMatMap.increment(
        getOrderedPair<BlockIndex>({blockSeq[edge.first], blockSeq[edge.second]}),
        counter
    );
}

void DegreeCorrectedStochasticBlockModelLikelihood::getDiffAdjMatMapFromEdgeMove(const Edge& edge, int counter, IntMap<std::pair<VertexIndex, VertexIndex>>& diffAdjMatMap) const{
    Edge orderedEdge = getOrderedEdge(edge);
    diffAdjMatMap.increment({orderedEdge.first, orderedEdge.second}, counter);
}


void DegreeCorrectedStochasticBlockModelLikelihood::getDiffEdgeMatMapFromBlockMove(
    const BlockMove& move, IntMap<std::pair<BlockIndex, BlockIndex>>& diffEdgeMatMap
) const {
    const BlockSequence& blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIndex)){
        BlockIndex blockIdx = blockSeq[neighbor.vertexIndex];
        size_t edgeMult = neighbor.label;
        std::pair<BlockIndex, BlockIndex> orderedBlockPair = getOrderedPair<BlockIndex> ({move.prevLabel, blockIdx});
        diffEdgeMatMap.decrement(orderedBlockPair, neighbor.label);

        if (neighbor.vertexIndex == move.vertexIndex) // handling self-loops
            blockIdx = move.nextLabel;

        orderedBlockPair = getOrderedPair<BlockIndex> ({move.nextLabel, blockIdx});
        diffEdgeMatMap.increment(orderedBlockPair, neighbor.label);
    }
}

const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihood() const {
    double logLikelihood = 0;
    const MultiGraph& edgeMat = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState() ;

    const CounterMap<size_t>& edgesInBlock = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getEdgeCounts();
    for (auto r : edgeMat){
        logLikelihood -= logFactorial(edgesInBlock[r]);
        for (auto s : edgeMat.getNeighboursOfIdx(r)){
            if (r <= s.vertexIndex)
                logLikelihood += (r == s.vertexIndex) ? logDoubleFactorial(2 * s.label) : logFactorial(s.label) ;
        }
    }

    const DegreeSequence& degreeSeq = (*m_degreePriorPtrPtr)->getState();
    for (auto v : *m_graphPtr){
        logLikelihood += logFactorial(degreeSeq[v]);
        for (auto n : m_graphPtr->getNeighboursOfIdx(v)){
            if (v <= n.vertexIndex)
                logLikelihood -= (v == n.vertexIndex) ? logDoubleFactorial(2 * n.label) : logFactorial(n.label);
        }
    }

    return logLikelihood;
}

const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const {
    const BlockSequence& blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
    const MultiGraph& edgeMat = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState();
    const CounterMap<size_t>& edgeCountsInBlocks = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getEdgeCounts();
    const CounterMap<size_t>& vertexCountsInBlocks = (*m_degreePriorPtrPtr)->getBlockPrior().getVertexCounts();
    double logLikelihoodRatioTerm = 0;

    IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
    IntMap<BlockIndex> diffEdgeCountsInBlocksMap;

    for (auto edge : move.addedEdges){
        getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
    }
    for (auto edge : move.removedEdges){
        getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);
    }

    for (auto diff : diffEdgeMatMap){
        auto r = diff.first.first, s = diff.first.second;
        auto ers = edgeMat.getEdgeMultiplicityIdx(r, s);
        diffEdgeCountsInBlocksMap.increment(r, diff.second);
        diffEdgeCountsInBlocksMap.increment(s, diff.second);
        logLikelihoodRatioTerm += (r == s)? logDoubleFactorial(2 * ers + 2 * diff.second) : logFactorial(ers + diff.second);
        logLikelihoodRatioTerm -= (r == s)? logDoubleFactorial(2 * ers) : logFactorial(ers);
    }

    for (auto diff : diffEdgeCountsInBlocksMap){
        logLikelihoodRatioTerm -= logFactorial(edgeCountsInBlocks[diff.first] + diff.second) ;
        logLikelihoodRatioTerm -= -logFactorial(edgeCountsInBlocks[diff.first]);
    }
    return logLikelihoodRatioTerm;
}

const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioAdjTerm (const GraphMove& move) const {
    IntMap<std::pair<VertexIndex, VertexIndex>> diffAdjMatMap;
    IntMap<VertexIndex> diffDegreeMap;
    double logLikelihoodRatioTerm = 0;


    for (auto edge : move.addedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, 1, diffAdjMatMap);
        diffDegreeMap.increment(edge.first);
        diffDegreeMap.increment(edge.second);
    }
    for (auto edge : move.removedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, -1, diffAdjMatMap);
        diffDegreeMap.decrement(edge.first);
        diffDegreeMap.decrement(edge.second);
    }

    for (auto diff : diffAdjMatMap){
        auto u = diff.first.first, v = diff.first.second;
        auto edgeMult = m_graphPtr->getEdgeMultiplicityIdx(u, v);
        logLikelihoodRatioTerm -= (u == v) ? logDoubleFactorial(2 * (edgeMult + diff.second)) : logFactorial(edgeMult + diff.second);
        logLikelihoodRatioTerm += (u == v) ? logDoubleFactorial(2 * edgeMult) : logFactorial(edgeMult);
    }

    const DegreeSequence& degreeSeq = (*m_degreePriorPtrPtr)->getState();
    for (auto diff : diffDegreeMap){
        logLikelihoodRatioTerm += logFactorial(degreeSeq[diff.first] + diff.second) - logFactorial(degreeSeq[diff.first]);
    }
    return logLikelihoodRatioTerm;
}

const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove (const BlockMove& move) const {
    const BlockSequence& blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
    const MultiGraph& edgeMat = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState();
    const CounterMap<size_t>& edgesInBlock = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getEdgeCounts();
    double logLikelihoodRatio = 0;

    if (move.prevLabel == move.nextLabel)
        return 0;

    IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
    IntMap<BlockIndex> diffEdgesInBlockMap;
    getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);
    for (auto diff : diffEdgeMatMap){
        size_t ers;
        auto r = diff.first.first, s = diff.first.second;
        auto dErs = diff.second;
        diffEdgesInBlockMap.increment(r, dErs);
        diffEdgesInBlockMap.increment(s, dErs);

        ers = (r < edgeMat.getSize() and s < edgeMat.getSize()) ? edgeMat.getEdgeMultiplicityIdx(r, s) : 0;
        logLikelihoodRatio += (r == s) ? logDoubleFactorial(2 * ers + 2 * dErs) : logFactorial(ers + dErs);
        logLikelihoodRatio -= (r == s) ? logDoubleFactorial(2 * ers) : logFactorial(ers);
    }

    for (auto diff : diffEdgesInBlockMap){
            auto r = diff.first;
            auto dEr = diff.second;
            size_t er = edgesInBlock[r];
            logLikelihoodRatio -= logFactorial(er + dEr) - logFactorial(er);
    }
    return logLikelihoodRatio;
}

}
