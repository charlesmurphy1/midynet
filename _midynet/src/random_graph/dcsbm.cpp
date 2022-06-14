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
    const EdgeMatrix& edgeMat = getEdgeMatrix();
    const DegreeSequence& degreeSeq = getDegrees();
    setGraph( generateDCSBM(blockSeq, edgeMat, degreeSeq) );
}

const double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihood() const{
    double logLikelihood = 0;

    const EdgeMatrix& edgeMat = getEdgeMatrix() ;
    const vector<size_t>& edgesInBlock = getEdgeCountsInBlocks();
    auto numBlocks = getBlockCount();
    for (size_t r = 0; r < numBlocks; r++) {
        logLikelihood += logDoubleFactorial(edgeMat[r][r]);
        logLikelihood -= logFactorial(edgesInBlock[r]);
        for (size_t s = r + 1; s < numBlocks; s++) {
            logLikelihood += logFactorial(edgeMat[r][s]);
        }
    }

    const DegreeSequence& degreeSeq = getDegrees();
    const MultiGraph& graph = getGraph();
    size_t neighborIdx, edgeMult;
    for (auto idx : graph){
        logLikelihood += logFactorial(degreeSeq[idx]);
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
            if (idx > neighborIdx){
                continue;
            }else if (idx == neighborIdx){
                logLikelihood -= logDoubleFactorial(2 * edgeMult);
            }else{
                logLikelihood -= logFactorial(edgeMult);
            }
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
    const EdgeMatrix& edgeMat = getEdgeMatrix();
    const vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();
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
        auto bu = diff.first.first, bv = diff.first.second;
        diffEdgeCountsInBlocksMap.increment(bu, diff.second);
        diffEdgeCountsInBlocksMap.increment(bv, diff.second);
        if (bu == bv){
            logLikelihoodRatioTerm += logDoubleFactorial(edgeMat[bu][bv] + 2 * diff.second) - logDoubleFactorial(edgeMat[bu][bv]);
        }
        else{
            logLikelihoodRatioTerm += logFactorial(edgeMat[bu][bv] + diff.second) - logFactorial(edgeMat[bu][bv]);
        }
    }

    for (auto diff : diffEdgeCountsInBlocksMap){
        logLikelihoodRatioTerm -= logFactorial(edgeCountsInBlocks[diff.first] + diff.second) - logFactorial(edgeCountsInBlocks[diff.first]) ;
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
        if (u == v){
            logLikelihoodRatioTerm -= logDoubleFactorial(2 * (edgeMult + diff.second)) - logDoubleFactorial(2 * edgeMult);
        }
        else{
            logLikelihoodRatioTerm -= logFactorial(edgeMult + diff.second) - logFactorial(edgeMult);
        }
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
    BlockSequence blockSeq = getBlocks();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgesInBlock = getEdgeCountsInBlocks();
    double logLikelihoodRatio = 0;

    IntMap<pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
    IntMap<BlockIndex> diffEdgesInBlockMap;
    getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);
    for (auto diff : diffEdgeMatMap){
        size_t ers;
        auto bu = diff.first.first, bv = diff.first.second;
        auto dErs = diff.second;
        diffEdgesInBlockMap.increment(bu, dErs);
        diffEdgesInBlockMap.increment(bv, dErs);

        if (bu == getBlockCount() or bv == getBlockCount()) ers = 0;
        else ers = edgeMat[bu][bv];
        if (bu == bv){
            logLikelihoodRatio += logDoubleFactorial(ers + 2 * dErs) - logDoubleFactorial(ers);
        }
        else{
            logLikelihoodRatio += logFactorial(ers + dErs) - logFactorial(ers);
        }
    }

    for (auto diff : diffEdgesInBlockMap){
            size_t er = 0;
            auto bu = diff.first;
            auto dEr = diff.second;
            if (bu == getBlockCount()) er = 0;
            else er = edgesInBlock[bu];
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

void DegreeCorrectedStochasticBlockModelFamily::_checkSelfConsistency() const{
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
