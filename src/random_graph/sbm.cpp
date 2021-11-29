#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;

void StochasticBlockModelFamily::sampleState(){
    auto blockSeq = getBlockSequence();
    auto edgeMat = getEdgeMatrix();
    auto graph = generateSBM(blockSeq, edgeMat);
    setState(graph);
}

void StochasticBlockModelFamily::samplePriors(){
    m_blockPrior.sample();
    m_edgeMatrixPrior.sample();
    computationFinished();
}

double StochasticBlockModelFamily::getLogLikelihood() const{
    double logLikelihood = 0;

    auto edgeMat = getEdgeMatrix() ;
    vector<size_t> edgeCountsInBlocks = getEdgeCountsInBlocks();
    vector<size_t> vertexCountsInBlocks = getVertexCountsInBlocks();

    auto numBlocks = edgeMat.size();
    for (size_t r = 0; r < numBlocks; r++) {
        logLikelihood += logDoubleFactorial(edgeMat[r][r]);
        logLikelihood -= edgeCountsInBlocks[r] * log(vertexCountsInBlocks[r]);
        for (size_t s = r + 1; s < numBlocks; s++) {
            logLikelihood += logFactorial(edgeMat[r][s]);
        }
    }

    auto graph = getState();
    size_t neighborIdx, edgeMult;
    for (auto idx : graph){
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

double StochasticBlockModelFamily::getLogPrior() {
    double logPrior = m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint();
    computationFinished();
};

double StochasticBlockModelFamily::getLogJoint() {
    return getLogLikelihood() + getLogPrior();
};

void StochasticBlockModelFamily::getDiffEdgeMatMapFromEdgeMove( const Edge& edge, int counter, map<pair<BlockIndex, BlockIndex>, int>& diffEdgeMatMap ){
    Edge orderedEdge = getOrderedEdge(edge);
    BlockSequence blockSeq = getBlockSequence();
    if (diffEdgeMatMap.count({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]}) == 0){
        diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, int>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.first]}, 0) );
    }

    diffEdgeMatMap[getOrderedPair<BlockIndex>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]})] += counter;
};

double StochasticBlockModelFamily::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) {
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgeCountsInBlocks = getEdgeCountsInBlocks();
    vector<size_t> vertexCountsInBlocks = getVertexCountsInBlocks();
    double logLikelihoodRatioTerm = 0;

    map<pair<BlockIndex, BlockIndex>, int> diffEdgeMatMap;
    map<BlockIndex, int> diffEdgeCountsInBlocksMap;

    for (auto edge : move.addedEdges){
        getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
    }
    for (auto edge : move.removedEdges){
        getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);
    }

    for (auto diff : diffEdgeMatMap){
        auto bu = diff.first.first, bv = diff.first.second;
        if (diffEdgeCountsInBlocksMap.count(bu) == 0) diffEdgeCountsInBlocksMap.insert(pair<BlockIndex, int>(bu, 0) );
        diffEdgeCountsInBlocksMap[bu] += diff.second;
        if (diffEdgeCountsInBlocksMap.count(bv) == 0) diffEdgeCountsInBlocksMap.insert(pair<BlockIndex, int>(bv, 0) );
        diffEdgeCountsInBlocksMap[bv] += diff.second;
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

void StochasticBlockModelFamily::getDiffAdjMatMapFromEdgeMove( const Edge& edge, int counter, map<pair<VertexIndex, VertexIndex>, int>& diffAdjMatMap){
    Edge orderedEdge = getOrderedEdge(edge);
    if (diffAdjMatMap.count({orderedEdge.first, orderedEdge.second}) == 0){
        diffAdjMatMap.insert( pair<pair<VertexIndex, VertexIndex>, int>({orderedEdge.first, orderedEdge.second}, 0) );
    }
    diffAdjMatMap[{orderedEdge.first, orderedEdge.second}] += counter;
};

double StochasticBlockModelFamily::getLogLikelihoodRatioAdjTerm (const GraphMove& move) {
    map<pair<VertexIndex, VertexIndex>, int> diffAdjMatMap;
    double logLikelihoodRatioTerm = 0;

    for (auto edge : move.addedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, 1, diffAdjMatMap);
    }
    for (auto edge : move.removedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, -1, diffAdjMatMap);
    }

    for (auto diff : diffAdjMatMap){
        auto u = diff.first.first, v = diff.first.second;
        auto edgeMult = m_state.getEdgeMultiplicityIdx(u, v);
        if (u == v){
            logLikelihoodRatioTerm -= logDoubleFactorial(2 * edgeMult + 2 * diff.second) - logDoubleFactorial(2 * edgeMult);
        }
        else{
            logLikelihoodRatioTerm -= logFactorial(edgeMult + diff.second) - logFactorial(edgeMult);
        }
    }
    return logLikelihoodRatioTerm;
};

double StochasticBlockModelFamily::getLogLikelihoodRatio (const GraphMove& move) {

    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

void StochasticBlockModelFamily::getDiffEdgeMatMapFromBlockMove( const BlockMove& move, map<pair<BlockIndex, BlockIndex>, int>& diffEdgeMatMap){
    BlockSequence blockSeq = getBlockSequence();
    for (auto neighbor : m_state.getNeighboursOfIdx(move.vertexIdx)){
        VertexIndex idx = neighbor.vertexIndex;
        size_t edgeMult = neighbor.label;
        pair<BlockIndex, BlockIndex> orderedBlockPair = getOrderedPair<BlockIndex> ({move.prevBlockIdx, blockSeq[idx]});
        if (diffEdgeMatMap.count(getOrderedPair<BlockIndex>({orderedBlockPair.first, orderedBlockPair.second})) == 0){
            diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, int>(orderedBlockPair, 0));
        }
        diffEdgeMatMap[orderedBlockPair] -= edgeMult;

        orderedBlockPair = getOrderedPair<BlockIndex> ({move.nextBlockIdx, blockSeq[idx]});
        if (diffEdgeMatMap.count(getOrderedPair<BlockIndex>({orderedBlockPair.first, orderedBlockPair.second})) == 0){
            diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, int>(orderedBlockPair, 0));
        }
        diffEdgeMatMap[orderedBlockPair] += edgeMult;
    }
};

double StochasticBlockModelFamily::getLogLikelihoodRatio(const BlockMove& move){
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgeCountsInBlocks = getEdgeCountsInBlocks();
    vector<size_t> verticesInBlock = getVertexCountsInBlocks();
    double logLikelihoodRatio = 0;

    map<pair<BlockIndex, BlockIndex>, int> diffEdgeMatMap;
    map<BlockIndex, int> diffEdgeCountsInBlocksMap;
    map<BaseGraph::VertexIndex, int> diffVertexCountsInBlocksMap;

    for (size_t r = 0; r < getBlockCount() + 1 ; ++r){
        diffVertexCountsInBlocksMap[r] = 0;
        diffEdgeCountsInBlocksMap[r] = 0;
    }


    diffVertexCountsInBlocksMap[move.prevBlockIdx] -= 1;
    diffVertexCountsInBlocksMap[move.nextBlockIdx] += 1;


    getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);

    for (auto diff : diffEdgeMatMap){
        auto bu = diff.first.first, bv = diff.first.second;
        diffEdgeCountsInBlocksMap[bu] += diff.second;
        diffEdgeCountsInBlocksMap[bv] += diff.second;
        if (bu == bv){
            logLikelihoodRatio += logDoubleFactorial(edgeMat[bu][bv] + 2 * diff.second) - logDoubleFactorial(edgeMat[bu][bv]);
        }
        else{
            logLikelihoodRatio += logFactorial(edgeMat[bu][bv] + diff.second) - logFactorial(edgeMat[bu][bv]);
        }
    }

    for (auto diff : diffEdgeCountsInBlocksMap){
        if (diff.first < getBlockCount()) {
            auto er = edgeCountsInBlocks[ diff.first ];
            auto nr = verticesInBlock[ diff.first ];
            logLikelihoodRatio -= (er + diff.second) * log(nr + diffVertexCountsInBlocksMap[diff.first]) - er * log(nr);
        } else if (diff.first == getBlockCount() && diff.second != 0){
            logLikelihoodRatio -= (diff.second) * log(diffVertexCountsInBlocksMap[diff.first]);
        }
    }

    return logLikelihoodRatio;
};

double StochasticBlockModelFamily::getLogPriorRatio (const GraphMove& move) {
    double logPriorRatio = m_blockPrior.getLogPriorRatioFromGraphMove(move) + m_edgeMatrixPrior.getLogPriorRatioFromGraphMove(move);
    computationFinished();
    return logPriorRatio;
};

double StochasticBlockModelFamily::getLogPriorRatio (const BlockMove& move) {
    double logPriorRatio = m_blockPrior.getLogPriorRatioFromBlockMove(move) + m_edgeMatrixPrior.getLogPriorRatioFromBlockMove(move);
    computationFinished();
    return logPriorRatio;
};

void StochasticBlockModelFamily::applyMove (const GraphMove& move){
    m_blockPrior.applyGraphMove(move);
    m_edgeMatrixPrior.applyGraphMove(move);
    RandomGraph::applyMove(move);
    computationFinished();
    #if DEBUG
    checkSelfConsistency();
    #endif
};
void StochasticBlockModelFamily::applyMove (const BlockMove& move){
    m_blockPrior.applyBlockMove(move);
    m_edgeMatrixPrior.applyBlockMove(move);
    computationFinished();
    #if DEBUG
    checkSelfConsistency();
    #endif
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

void StochasticBlockModelFamily::checkSelfConsistency(){
    m_blockPrior.checkSelfConsistency();
    m_edgeMatrixPrior.checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_state, getBlockSequence(), getEdgeMatrix());
}
