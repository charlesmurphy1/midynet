#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;

void StochasticBlockModelFamily::sampleGraph(){
    const BlockSequence& blockSeq = getBlocks();
    const EdgeMatrix& edgeMat = getEdgeMatrix();
    MultiGraph graph = generateSBM(blockSeq, edgeMat);
    setGraph(graph);
}

void StochasticBlockModelFamily::samplePriors(){
    m_blockPriorPtr->sample();
    m_edgeMatrixPriorPtr->sample();
    computationFinished();
}

const double StochasticBlockModelFamily::getLogLikelihood() const{
    double edgePart = 0;

    const EdgeMatrix& edgeMat = getEdgeMatrix() ;
    const vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();

    const size_t& numBlocks = edgeMat.size();
    for (size_t r = 0; r < numBlocks; r++) {
        if (vertexCountsInBlocks[r] == 0)
            continue;
        edgePart += logDoubleFactorial(edgeMat[r][r]);
        edgePart -= edgeCountsInBlocks[r] * log(vertexCountsInBlocks[r]);
        for (size_t s = r + 1; s < numBlocks; s++) {
            edgePart += logFactorial(edgeMat[r][s]);
        }
    }
    const MultiGraph& graph = getGraph();
    size_t neighborIdx, edgeMult;
    double adjPart = 0;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
            if (idx > neighborIdx){
                continue;
            }
            if (idx == neighborIdx){
                adjPart -= logDoubleFactorial(2 * edgeMult);
            }else{
                adjPart -= logFactorial(edgeMult);
            }
        }
    }

    return edgePart + adjPart;
};

const double StochasticBlockModelFamily::getLogPrior() const {
    double logPrior = m_blockPriorPtr->getLogJoint() + m_edgeMatrixPriorPtr->getLogJoint();
    computationFinished();
    return logPrior;
};

void StochasticBlockModelFamily::getDiffEdgeMatMapFromEdgeMove( const Edge& edge, int counter, IntMap<pair<BlockIndex, BlockIndex>>& diffEdgeMatMap ) const{
    Edge orderedEdge = getOrderedEdge(edge);
    const BlockSequence& blockSeq = getBlocks();
    diffEdgeMatMap.increment(getOrderedPair<BlockIndex>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]}), counter);
};

const double StochasticBlockModelFamily::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const {
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
            logLikelihoodRatioTerm -= diff.second *  log( vertexCountsInBlocks[diff.first] ) ;
    }
    return logLikelihoodRatioTerm;
};

void StochasticBlockModelFamily::getDiffAdjMatMapFromEdgeMove(
    const Edge& edge, int counter, IntMap<pair<VertexIndex, VertexIndex>>& diffAdjMatMap
) const{
    Edge orderedEdge = getOrderedEdge(edge);
    diffAdjMatMap.increment({orderedEdge.first, orderedEdge.second}, counter);
};

const double StochasticBlockModelFamily::getLogLikelihoodRatioAdjTerm (const GraphMove& move) const {
    IntMap<pair<VertexIndex, VertexIndex>> diffAdjMatMap;
    double logLikelihoodRatioTerm = 0;

    for (auto edge : move.addedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, 1, diffAdjMatMap);
    }
    for (auto edge : move.removedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, -1, diffAdjMatMap);
    }

    for (auto diff : diffAdjMatMap){
        auto u = diff.first.first, v = diff.first.second;
        auto edgeMult = m_graph.getEdgeMultiplicityIdx(u, v);
        if (u == v){
            logLikelihoodRatioTerm -= logDoubleFactorial(2 * edgeMult + 2 * diff.second) - logDoubleFactorial(2 * edgeMult);
        }
        else{
            logLikelihoodRatioTerm -= logFactorial(edgeMult + diff.second) - logFactorial(edgeMult);
        }
    }
    return logLikelihoodRatioTerm;
};

const double StochasticBlockModelFamily::getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

void StochasticBlockModelFamily::getDiffEdgeMatMapFromBlockMove(
    const BlockMove& move, IntMap<pair<BlockIndex, BlockIndex>>& diffEdgeMatMap
) const {
    const BlockSequence& blockSeq = getBlocks();
    for (auto neighbor : m_graph.getNeighboursOfIdx(move.vertexIdx)){
        BlockIndex blockIdx = blockSeq[neighbor.vertexIndex];
        size_t edgeMult = neighbor.label;
        pair<BlockIndex, BlockIndex> orderedBlockPair = getOrderedPair<BlockIndex> ({move.prevBlockIdx, blockIdx});
        diffEdgeMatMap.decrement(orderedBlockPair, neighbor.label);

        if (neighbor.vertexIndex == move.vertexIdx) // handling self-loops
            blockIdx = move.nextBlockIdx;

        orderedBlockPair = getOrderedPair<BlockIndex> ({move.nextBlockIdx, blockIdx});
        diffEdgeMatMap.increment(orderedBlockPair, neighbor.label);
    }
};

const double StochasticBlockModelFamily::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    const BlockSequence& blockSeq = getBlocks();
    const EdgeMatrix& edgeMat = getEdgeMatrix();
    const vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const vector<size_t>& verticesInBlock = getVertexCountsInBlocks();
    double logLikelihoodRatio = 0;

    IntMap<pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
    IntMap<BlockIndex> diffEdgeCountsInBlocksMap;
    IntMap<BaseGraph::VertexIndex> diffVertexCountsInBlocksMap;

    diffVertexCountsInBlocksMap.decrement(move.prevBlockIdx);
    diffVertexCountsInBlocksMap.increment(move.nextBlockIdx);


    getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);

    for (auto diff : diffEdgeMatMap){
        auto bu = diff.first.first, bv = diff.first.second;
        diffEdgeCountsInBlocksMap.increment(bu, diff.second);
        diffEdgeCountsInBlocksMap.increment(bv, diff.second);
        size_t ers;
        if (bu == getBlockCount() or bv == getBlockCount()) ers = 0;
        else ers = edgeMat[bu][bv];
        if (bu == bv){
            logLikelihoodRatio += logDoubleFactorial(ers + 2 * diff.second) - logDoubleFactorial(ers);
        }
        else{
            logLikelihoodRatio += logFactorial(ers + diff.second) - logFactorial(ers);

        }
    }

    for (size_t r = 0; r <= getBlockCount(); ++r){
        int dEr = diffEdgeCountsInBlocksMap[r];
        int dNr = diffVertexCountsInBlocksMap[r];
        if (r < getBlockCount()) {
            auto er = edgeCountsInBlocks[r];
            auto nr = verticesInBlock[r];
            if (er + dEr == 0 && nr + dNr == 0)
                logLikelihoodRatio -= -er * log(nr);
            else
                logLikelihoodRatio -= (er + dEr) * log(nr + dNr) - er * log(nr);
        } else if (r == getBlockCount() && dEr != 0){
            logLikelihoodRatio -= (dEr) * log(dNr);
        }
    }

    return logLikelihoodRatio;
};

const double StochasticBlockModelFamily::getLogPriorRatioFromGraphMove (const GraphMove& move) const {
    double logPriorRatio = m_blockPriorPtr->getLogJointRatioFromGraphMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move);
    // computationFinished();
    return logPriorRatio;
};

const double StochasticBlockModelFamily::getLogPriorRatioFromBlockMove (const BlockMove& move) const {
    double logPriorRatio = m_blockPriorPtr->getLogJointRatioFromBlockMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromBlockMove(move);
    // computationFinished();
    return logPriorRatio;
};

void StochasticBlockModelFamily::_applyGraphMove (const GraphMove& move) {
    m_blockPriorPtr->applyGraphMove(move);
    m_edgeMatrixPriorPtr->applyGraphMove(move);
    RandomGraph::_applyGraphMove(move);
};

void StochasticBlockModelFamily::_applyBlockMove (const BlockMove& move){
    m_blockPriorPtr->applyBlockMove(move);
    m_edgeMatrixPriorPtr->applyBlockMove(move);
};

EdgeMatrix StochasticBlockModelFamily::getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    EdgeMatrix edgeMat(numBlocks, vector<size_t>(numBlocks, 0));
    size_t neighborIdx, edgeMult, r, s;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
            r = blockSeq[idx];
            s = blockSeq[neighborIdx];

            if (idx == neighbor.vertexIndex){
                edgeMat[s][r] += 2 * neighbor.label;
                edgeMat[r][s] += 2 * neighbor.label;
            }else{
                edgeMat[s][r] += neighbor.label;
                edgeMat[r][s] += neighbor.label;
            }
        }
    }
    for (auto r = 0; r < numBlocks; r ++){
        for (auto s = 0; s < numBlocks; s ++){
            edgeMat[r][s] /= 2; // all edges are counted twice.
        }
    }
    return edgeMat;
};

void StochasticBlockModelFamily::checkGraphConsistencyWithEdgeMatrix(
    const MultiGraph& graph,
    const BlockSequence& blockSeq,
    const EdgeMatrix& expectedEdgeMat){
    EdgeMatrix actualEdgeMat = getEdgeMatrixFromGraph(graph, blockSeq);
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    for (auto r = 0; r < numBlocks; ++r){
        for (auto s = 0; s < numBlocks; ++s){
            if (expectedEdgeMat[r][s] != actualEdgeMat[r][s])
                throw ConsistencyError("StochasticBlockModelFamily: at indices ("
                + to_string(r) + ", " + to_string(s) + ") edge matrix is inconsistent with graph:"
                + to_string(expectedEdgeMat[r][s]) + " != " + to_string(actualEdgeMat[r][s]));
        }
    }


};

void StochasticBlockModelFamily::checkSelfConsistency() const{
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeMatrixPriorPtr->checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_graph, getBlocks(), getEdgeMatrix());
}

void StochasticBlockModelFamily::checkSelfSafety()const{
    if (m_blockPriorPtr == nullptr)
        throw SafetyError("StochasticBlockModelFamily: unsafe family since `m_blockPriorPtr` is empty.");
    m_blockPriorPtr->checkSafety();

    if (m_edgeMatrixPriorPtr == nullptr)
        throw SafetyError("StochasticBlockModelFamily: unsafe family since `m_edgeMatrixPriorPtr` is empty.");
    m_edgeMatrixPriorPtr->checkSafety();
}
