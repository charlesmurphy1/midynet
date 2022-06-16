#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;

void DegreeCorrectedStochasticBlockModelFamily::samplePriors(){
    m_blockPriorPtr->sample();
    m_edgeMatrixPriorPtr->sample();
    m_degreePriorPtr->sample();
    computationFinished();
};

void DegreeCorrectedStochasticBlockModelFamily::sampleGraph(){
    const BlockSequence& blockSeq = getBlocks();
    const MultiGraph& edgeMat = getEdgeMatrix();
    const DegreeSequence& degreeSeq = getDegrees();
    setGraph( generateDCSBM(blockSeq, edgeMat.getAdjacencyMatrix(), degreeSeq) );
}

const double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihood() const{
    double logLikelihood = 0;

    const MultiGraph& edgeMat = getEdgeMatrix() ;
    const CounterMap<size_t>& edgesInBlock = getEdgeCountsInBlocks();
    for (auto r : edgeMat){
        logLikelihood -= logFactorial(edgesInBlock[r]);
        for (auto s : edgeMat.getNeighboursOfIdx(r)){
            if (r <= s.vertexIndex)
                logLikelihood += (r == s.vertexIndex) ? logDoubleFactorial(2 * s.label) : logFactorial(s.label) ;
        }
    }

    const DegreeSequence& degreeSeq = getDegrees();
    const MultiGraph& graph = getGraph();
    for (auto v : graph){
        logLikelihood += logFactorial(degreeSeq[v]);
        for (auto n : graph.getNeighboursOfIdx(v)){
            if (v <= n.vertexIndex)
                logLikelihood -= (v == n.vertexIndex) ? logDoubleFactorial(2 * n.label) : logFactorial(n.label);
        }
    }

    return logLikelihood;
};

const double DegreeCorrectedStochasticBlockModelFamily::getLogPrior() const {
    double logPrior =  m_blockPriorPtr->getLogJoint() + m_edgeMatrixPriorPtr->getLogJoint() + m_degreePriorPtr->getLogJoint();
    computationFinished();
    return logPrior;
};

const double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const {
    const BlockSequence& blockSeq = getBlocks();
    const MultiGraph& edgeMat = getEdgeMatrix();
    const CounterMap<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const CounterMap<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();
    double logLikelihoodRatioTerm = 0;

    IntMap<pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
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
};

const double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioAdjTerm (const GraphMove& move) const {
    IntMap<pair<VertexIndex, VertexIndex>> diffAdjMatMap;
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
        auto edgeMult = m_graph.getEdgeMultiplicityIdx(u, v);
        logLikelihoodRatioTerm -= (u == v) ? logDoubleFactorial(2 * (edgeMult + diff.second)) : logFactorial(edgeMult + diff.second);
        logLikelihoodRatioTerm += (u == v) ? logDoubleFactorial(2 * edgeMult) : logFactorial(edgeMult);
    }

    const DegreeSequence& degreeSeq = getDegrees();
    for (auto diff : diffDegreeMap){
        logLikelihoodRatioTerm += logFactorial(degreeSeq[diff.first] + diff.second) - logFactorial(degreeSeq[diff.first]);
    }
    return logLikelihoodRatioTerm;
};


const double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

const double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    const BlockSequence& blockSeq = getBlocks();
    const MultiGraph& edgeMat = getEdgeMatrix();
    const CounterMap<size_t>& edgesInBlock = getEdgeCountsInBlocks();
    double logLikelihoodRatio = 0;

    if (move.prevBlockIdx == move.nextBlockIdx)
        return 0;

    IntMap<pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
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
            size_t er = 0;
            auto r = diff.first;
            auto dEr = diff.second;
            if (r == getBlockCount()) er = 0;
            else er = edgesInBlock[r];
            logLikelihoodRatio -= logFactorial(er + dEr) - logFactorial(er);
    }
    return logLikelihoodRatio;
};

const double DegreeCorrectedStochasticBlockModelFamily::getLogPriorRatioFromGraphMove(const GraphMove& move) const {
    double logPriorRatio = m_blockPriorPtr->getLogJointRatioFromGraphMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move) + m_degreePriorPtr->getLogJointRatioFromGraphMove(move);
    computationFinished();
    return logPriorRatio;
}

const double DegreeCorrectedStochasticBlockModelFamily::getLogPriorRatioFromBlockMove(const BlockMove& move) const {
    double logPriorRatio = m_blockPriorPtr->getLogJointRatioFromBlockMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromBlockMove(move) + m_degreePriorPtr->getLogJointRatioFromBlockMove(move);
    computationFinished();
    return logPriorRatio;
}

void DegreeCorrectedStochasticBlockModelFamily::_applyGraphMove (const GraphMove& move) {
    m_blockPriorPtr->applyGraphMove(move);
    m_edgeMatrixPriorPtr->applyGraphMove(move);
    m_degreePriorPtr->applyGraphMove(move);
    RandomGraph::_applyGraphMove(move);
};
void DegreeCorrectedStochasticBlockModelFamily::_applyBlockMove (const BlockMove& move){
    m_blockPriorPtr->applyBlockMove(move);
    m_edgeMatrixPriorPtr->applyBlockMove(move);
    m_degreePriorPtr->applyBlockMove(move);
};

DegreeSequence DegreeCorrectedStochasticBlockModelFamily::getDegreesFromGraph(const MultiGraph& graph){
    DegreeSequence degreeSeq(graph.getSize(), 0);

    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            if (idx == neighbor.vertexIndex) degreeSeq[idx] += 2 * neighbor.label;
            else degreeSeq[idx] += neighbor.label;
        }
    }

    return degreeSeq;
}

void DegreeCorrectedStochasticBlockModelFamily::checkGraphConsistencyWithDegreeSequence(const MultiGraph& graph, const DegreeSequence& expectedDegreeSeq){
    DegreeSequence actualDegreeSeq = getDegreesFromGraph(graph);

    for (auto idx : graph){
        if (expectedDegreeSeq[idx] != actualDegreeSeq[idx])
            throw ConsistencyError("DCSBMFamily: expected degree of index " + to_string(idx)
            + " is inconsistent with graph : " + to_string(expectedDegreeSeq[idx]) + " != " + to_string(actualDegreeSeq[idx]));
    }
}

void DegreeCorrectedStochasticBlockModelFamily::checkSelfConsistency() const{
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeMatrixPriorPtr->checkSelfConsistency();
    m_degreePriorPtr->checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_graph, getBlocks(), getEdgeMatrix());
    checkGraphConsistencyWithDegreeSequence(m_graph, getDegrees());
}

void DegreeCorrectedStochasticBlockModelFamily::checkSelfSafety()const{
    StochasticBlockModelFamily::checkSafety();
    if (m_degreePriorPtr == nullptr)
        throw SafetyError("StochasticBlockModelFamily: unsafe family since `m_degreePriorPtr` is empty.");
    m_degreePriorPtr->checkSafety();
}
