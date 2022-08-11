#include "FastMIDyNet/random_graph/likelihood/sbm.h"
#include "BaseGraph/types.h"

using namespace BaseGraph;
namespace FastMIDyNet{

void StochasticBlockModelLikelihood::getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge& edge, int counter, IntMap<std::pair<BlockIndex, BlockIndex>>& diffEdgeMatMap) const{
    const BlockSequence& blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
    diffEdgeMatMap.increment(
        getOrderedPair<BlockIndex>({blockSeq[edge.first], blockSeq[edge.second]}),
        counter
    );
}

void StochasticBlockModelLikelihood::getDiffAdjMatMapFromEdgeMove(const Edge& edge, int counter, IntMap<std::pair<VertexIndex, VertexIndex>>& diffAdjMatMap) const{
    Edge orderedEdge = getOrderedEdge(edge);
    diffAdjMatMap.increment({orderedEdge.first, orderedEdge.second}, counter);
}


void StochasticBlockModelLikelihood::getDiffEdgeMatMapFromBlockMove(
    const BlockMove& move, IntMap<std::pair<BlockIndex, BlockIndex>>& diffEdgeMatMap
) const {
    const BlockSequence& blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
    for (auto neighbor : m_statePtr->getNeighboursOfIdx(move.vertexIndex)){
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

const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const {
    const BlockSequence& blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
    const MultiGraph& labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
    const CounterMap<size_t>& edgeCounts = (*m_labelGraphPriorPtrPtr)->getEdgeCounts();
    const CounterMap<size_t>& vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
    double logLikelihoodRatioTerm = 0;

    IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
    IntMap<BlockIndex> diffEdgeCountsMap;

    for (auto edge : move.addedEdges)
        getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
    for (auto edge : move.removedEdges)
        getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);

    for (auto diff : diffEdgeMatMap){
        auto r = diff.first.first, s = diff.first.second;
        size_t ers = (r >= labelGraph.getSize() or s >= labelGraph.getSize()) ? 0 : labelGraph.getEdgeMultiplicityIdx(r, s);
        diffEdgeCountsMap.increment(r, diff.second);
        diffEdgeCountsMap.increment(s, diff.second);
        logLikelihoodRatioTerm += (r == s) ? logDoubleFactorial(2 * ers + 2 * diff.second) : logFactorial(ers + diff.second);
        logLikelihoodRatioTerm -= (r == s) ? logDoubleFactorial(2 * ers) : logFactorial(ers);
    }

    for (auto diff : diffEdgeCountsMap){
            logLikelihoodRatioTerm -= diff.second * log( vertexCounts[diff.first] ) ;
    }
    return logLikelihoodRatioTerm;
}

const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioAdjTerm (const GraphMove& move) const {
    IntMap<std::pair<VertexIndex, VertexIndex>> diffAdjMatMap;
    double logLikelihoodRatioTerm = 0;

    for (auto edge : move.addedEdges)
        getDiffAdjMatMapFromEdgeMove(edge, 1, diffAdjMatMap);
    for (auto edge : move.removedEdges)
        getDiffAdjMatMapFromEdgeMove(edge, -1, diffAdjMatMap);

    for (auto diff : diffAdjMatMap){
        auto u = diff.first.first, v = diff.first.second;
        auto edgeMult = m_statePtr->getEdgeMultiplicityIdx(u, v);
        logLikelihoodRatioTerm -= (u == v) ? logDoubleFactorial(2 * edgeMult + 2 * diff.second) : logFactorial(edgeMult + diff.second);
        logLikelihoodRatioTerm += (u == v) ? logDoubleFactorial(2 * edgeMult) : logFactorial(edgeMult);
    }
    return logLikelihoodRatioTerm;
}

const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihood() const {

    const MultiGraph& labelGraph = (*m_labelGraphPriorPtrPtr)->getState() ;
    const CounterMap<size_t>& vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();

    double logLikelihood = 0;

    for (const auto& r : labelGraph) {
        auto er = labelGraph.getDegreeOfIdx(r), nr = vertexCounts.get(r);
        if (er == 0 or vertexCounts.get(r) == 0)
            continue;
        logLikelihood -= er * log(nr);
        for (const auto& s : labelGraph.getNeighboursOfIdx(r))
            if (r <= s.vertexIndex)
                logLikelihood += (r == s.vertexIndex) ? logDoubleFactorial(2 * s.label) : logFactorial(s.label);
    }
    for (const auto& idx : *m_statePtr)
        for (const auto& neighbor : m_statePtr->getNeighboursOfIdx(idx))
            if (idx <= neighbor.vertexIndex)
                logLikelihood -= (idx == neighbor.vertexIndex) ? logDoubleFactorial(2 * neighbor.label) : logFactorial(neighbor.label);
    return logLikelihood;
}

const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove (const BlockMove& move) const {
    if (move.prevLabel == move.nextLabel or move.level > 0)
        return 0;
    const BlockSequence& blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
    const MultiGraph& edgeMat = (*m_labelGraphPriorPtrPtr)->getState();
    const CounterMap<size_t>& edgeCounts = (*m_labelGraphPriorPtrPtr)->getEdgeCounts();
    const CounterMap<size_t>& vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
    const size_t& degree = m_statePtr->getDegreeOfIdx(move.vertexIndex);
    double logLikelihoodRatio = 0;


    IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;

    getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);

    for (auto diff : diffEdgeMatMap){
        auto r = diff.first.first, s = diff.first.second;
        size_t ers = (r >= edgeMat.getSize() or s >= edgeMat.getSize()) ? 0 : edgeMat.getEdgeMultiplicityIdx(r, s);
        logLikelihoodRatio += (r == s) ? logDoubleFactorial(2 * ers + 2 * diff.second) : logFactorial(ers + diff.second);
        logLikelihoodRatio -= (r == s) ? logDoubleFactorial(2 * ers) : logFactorial(ers);
    }

    logLikelihoodRatio += edgeCounts[move.prevLabel] * log(vertexCounts[move.prevLabel]) ;
    logLikelihoodRatio -= (vertexCounts.get(move.prevLabel) == 1) ? 0: (edgeCounts[move.prevLabel] - degree) * log(vertexCounts[move.prevLabel] - 1);

    logLikelihoodRatio += (vertexCounts.get(move.nextLabel) == 0) ? 0: edgeCounts[move.nextLabel] * log(vertexCounts[move.nextLabel]);
    logLikelihoodRatio -= (edgeCounts[move.nextLabel] + degree) * log(vertexCounts[move.nextLabel] + 1) ;
    return logLikelihoodRatio;
}

const double UniformStochasticBlockModelLikelihood::getLogLikelihood() const {

    const MultiGraph& labelGraph = (*m_labelGraphPriorPtrPtr)->getState() ;
    const CounterMap<size_t>& vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
    const auto& likelihoodFunction = (*m_withParallelEdgesPtr) ? logMultisetCoefficient: logBinomialCoefficient;


    double logLikelihood = 0;

    for (const auto& r : labelGraph) {
        if (vertexCounts[r] == 0)
            continue;
        if (vFunc(vertexCounts[r]) < labelGraph.getEdgeMultiplicityIdx(r, r) and not *m_withParallelEdgesPtr)
            return -INFINITY;
        logLikelihood -= likelihoodFunction(
            vFunc(vertexCounts[r]),
            labelGraph.getEdgeMultiplicityIdx(r, r)
        ) ;
        for (const auto& s : labelGraph.getNeighboursOfIdx(r))
            if (r < s.vertexIndex){
                if (vertexCounts[r] * vertexCounts[s.vertexIndex] < s.label and not *m_withParallelEdgesPtr)
                    return -INFINITY;
                logLikelihood -= likelihoodFunction(vertexCounts[r] * vertexCounts[s.vertexIndex], s.label);
            }
    }
    return logLikelihood;
}

const double UniformStochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
    const MultiGraph& labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
    const CounterMap<size_t>& vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
    const auto& likelihoodFunction = (*m_withParallelEdgesPtr) ? logMultisetCoefficient: logBinomialCoefficient;
    double logLikelihoodRatio = 0;

    IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;

    for (auto edge : move.addedEdges)
        getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
    for (auto edge : move.removedEdges)
        getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);

    for (auto diff : diffEdgeMatMap){
        auto r = diff.first.first, s = diff.first.second;
        size_t nr = vertexCounts[r], ns = vertexCounts[s];
        int ers = (r >= labelGraph.getSize() or s >= labelGraph.getSize()) ? 0 : labelGraph.getEdgeMultiplicityIdx(r, s);
        int vertexTerm = (r == s) ? vFunc(nr) : nr * ns;
        logLikelihoodRatio -= likelihoodFunction(vertexTerm, ers + diff.second);
        logLikelihoodRatio += likelihoodFunction(vertexTerm, ers);
    }

    return logLikelihoodRatio;
}

const double UniformStochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove (const BlockMove& move) const {
    if (move.prevLabel == move.nextLabel or move.level > 0)
        return 0;
    const MultiGraph& labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
    const CounterMap<size_t>& vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
    const auto& likelihoodFunction = (*m_withParallelEdgesPtr) ? logMultisetCoefficient: logBinomialCoefficient;

    double logLikelihoodRatio = 0;

    IntMap<BlockIndex> vDiffMap;
    vDiffMap.decrement(move.prevLabel);
    vDiffMap.increment(move.nextLabel);

    IntMap<std::pair<BlockIndex, BlockIndex>> eDiffMap;
    getDiffEdgeMatMapFromBlockMove(move, eDiffMap);
    for (auto diff : eDiffMap){
        auto r = diff.first.first, s = diff.first.second;
        size_t nr = vertexCounts[r], ns = vertexCounts[s];
        int edgeTermBefore = (r >= labelGraph.getSize() or s >= labelGraph.getSize()) ? 0 : labelGraph.getEdgeMultiplicityIdx(r, s);
        int edgeTermAfter = (r >= labelGraph.getSize() or s >= labelGraph.getSize()) ? diff.second : (labelGraph.getEdgeMultiplicityIdx(r, s) + diff.second);
        int vertexTermBefore = (r == s) ? vFunc(nr) : nr * ns;
        int vertexTermAfter = (r == s) ? vFunc(nr + vDiffMap.get(r)) : (nr + vDiffMap.get(r)) * (ns + vDiffMap.get(s));
        if (vertexTermAfter < edgeTermAfter and not *m_withParallelEdgesPtr)
            return -INFINITY;
        logLikelihoodRatio -= likelihoodFunction(vertexTermAfter, edgeTermAfter) - likelihoodFunction(vertexTermBefore, edgeTermBefore);
    }

    // remaining contributions that did not change the edge counts
    std::set<BaseGraph::Edge> visited;
    for (const auto& diff : vDiffMap){
        if (diff.first >= labelGraph.getSize())
            continue;
        for (const auto& neighbor : labelGraph.getNeighboursOfIdx(diff.first)){
            BlockIndex r = diff.first, s = neighbor.vertexIndex;
            // if not empty, the term has been processed in the edgeDiff loop
            auto rs = getOrderedEdge({r, s});
            if (visited.count(rs) > 0 or not eDiffMap.isEmpty(rs))
                continue;
            visited.insert(rs);
            size_t nr = vertexCounts[r], ns = vertexCounts[s];
            int vertexTermBefore = (r == s) ? vFunc(nr) : nr * ns;
            int vertexTermAfter = (r == s) ? vFunc(nr + vDiffMap.get(r)) : (nr + vDiffMap.get(r)) * (ns + vDiffMap.get(s));
            int edgeTerm = neighbor.label;
            if (vertexTermAfter < edgeTerm and not *m_withParallelEdgesPtr)
                return -INFINITY;
            logLikelihoodRatio -= likelihoodFunction(vertexTermAfter, edgeTerm) - likelihoodFunction(vertexTermBefore, edgeTerm);
        }
    }


    return logLikelihoodRatio;
}

}
