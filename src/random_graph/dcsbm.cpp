#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;

void DegreeCorrectedStochasticBlockModelFamily::samplePriors(){
    m_blockPrior.sample();
    m_edgeMatrixPrior.sample();
    m_degreePrior.sample();
    computationFinished();
};

void DegreeCorrectedStochasticBlockModelFamily::sampleState(){
    const BlockSequence& blockSeq = getBlockSequence();
    const EdgeMatrix& edgeMat = getEdgeMatrix();
    const DegreeSequence& degreeSeq = getDegreeSequence();
    setState( generateDCSBM(blockSeq, edgeMat, degreeSeq) );
}

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihood() const{
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

    const DegreeSequence& degreeSeq = getDegreeSequence();
    const MultiGraph& graph = getState();
    size_t neighborIdx, edgeMult;
    for (auto idx : graph){
        logLikelihood += logFactorial(degreeSeq[idx]);
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
            if (idx < neighborIdx){
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

double DegreeCorrectedStochasticBlockModelFamily::getLogPrior() {
    double logPrior =  m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint() + m_degreePrior.getLogJoint();
    computationFinished();
    return logPrior;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) {
    const BlockSequence& blockSeq = getBlockSequence();
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

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioAdjTerm (const GraphMove& move) {
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
        auto edgeMult = m_state.getEdgeMultiplicityIdx(u, v);
        if (u == v){
            logLikelihoodRatioTerm -= logDoubleFactorial(2 * (edgeMult + diff.second)) - logDoubleFactorial(2 * edgeMult);
        }
        else{
            logLikelihoodRatioTerm -= logFactorial(edgeMult + diff.second) - logFactorial(edgeMult);
        }
    }

    const DegreeSequence& degreeSeq = getDegreeSequence();
    for (auto diff : diffDegreeMap){
        logLikelihoodRatioTerm += logFactorial(degreeSeq[diff.first] + diff.second) - logFactorial(degreeSeq[diff.first]);
    }
    return logLikelihoodRatioTerm;
};


double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio (const GraphMove& move) {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio(const BlockMove& move){
    BlockSequence blockSeq = getBlockSequence();
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

double DegreeCorrectedStochasticBlockModelFamily::getLogPriorRatio(const GraphMove& move){
    double logPriorRatio = m_blockPrior.getLogPriorRatioFromGraphMove(move) + m_edgeMatrixPrior.getLogPriorRatioFromGraphMove(move) + m_degreePrior.getLogPriorRatioFromGraphMove(move);
    computationFinished();
    return logPriorRatio;
}

double DegreeCorrectedStochasticBlockModelFamily::getLogPriorRatio(const BlockMove& move){
    double logPriorRatio = m_blockPrior.getLogPriorRatioFromBlockMove(move) + m_edgeMatrixPrior.getLogPriorRatioFromBlockMove(move) + m_degreePrior.getLogPriorRatioFromBlockMove(move);
    computationFinished();
    return logPriorRatio;
}

void DegreeCorrectedStochasticBlockModelFamily::applyMove (const GraphMove& move){
    m_blockPrior.applyGraphMove(move);
    m_edgeMatrixPrior.applyGraphMove(move);
    m_degreePrior.applyGraphMove(move);
    RandomGraph::applyMove(move);
    computationFinished();
    #if DEBUG
    checkSelfConsistency();
    #endif
};
void DegreeCorrectedStochasticBlockModelFamily::applyMove (const BlockMove& move){
    m_blockPrior.applyBlockMove(move);
    m_edgeMatrixPrior.applyBlockMove(move);
    m_degreePrior.applyBlockMove(move);
    computationFinished();
    #if DEBUG
    checkSelfConsistency();
    #endif
};

DegreeSequence DegreeCorrectedStochasticBlockModelFamily::getDegreeSequenceFromGraph(const MultiGraph& graph){
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
    DegreeSequence actualDegreeSeq = getDegreeSequenceFromGraph(graph);

    for (auto idx : graph){
        if (expectedDegreeSeq[idx] != actualDegreeSeq[idx])
            throw ConsistencyError("DCSBMFamily: expected degree of index " + to_string(idx)
            + " is inconsistent with graph : " + to_string(expectedDegreeSeq[idx]) + " != " + to_string(actualDegreeSeq[idx]));
    }
}

void DegreeCorrectedStochasticBlockModelFamily::checkSelfConsistency(){
    m_blockPrior.checkSelfConsistency();
    m_edgeMatrixPrior.checkSelfConsistency();
    m_degreePrior.checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_state, getBlockSequence(), getEdgeMatrix());
    checkGraphConsistencyWithDegreeSequence(m_state, getDegreeSequence());
}
