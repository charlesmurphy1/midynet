#include <algorithm>
#include <map>
#include <utility>
#include <vector>

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

double StochasticBlockModelFamily::getLogLikelihood() const{
    double logLikelihood = 0;

    auto edgeMat = getEdgeMatrix() ;
    vector<size_t> edgesInBlock = getEdgeCountsInBlock();
    vector<size_t> vertexCountsInBlock = getVertexCountsInBlock();

    auto numBlocks = edgeMat.size();
    for (size_t r = 0; r < numBlocks; r++) {
        logLikelihood += logDoubleFactorial(edgeMat[r][r]);
        logLikelihood -= edgesInBlock[r] * log(vertexCountsInBlock[r]);
        for (size_t s = r + 1; s < numBlocks; s++) {
            logLikelihood += logFactorial(edgeMat[r][s]);
        }
    }

    auto graph = getState();
    size_t neighborIdx, edgeMult;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.first;
            edgeMult = neighbor.second;
            if (idx < neighborIdx){
                continue;
            }else if (idx == neighborIdx){
                logLikelihood -= logDoubleFactorial(edgeMult);
            }else{
                logLikelihood -= logFactorial(edgeMult);
            }
        }
    }

    return logLikelihood;
};

double StochasticBlockModelFamily::getLogPrior() {
    return m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint();
};

double StochasticBlockModelFamily::getLogJoint() {
    return getLogLikelihood() + getLogPrior();
};

void StochasticBlockModelFamily::getDiffEdgeMatMapFromEdgeMove( const Edge& edge, int counter, map<pair<BlockIndex, BlockIndex>, size_t>& diffEdgeMatMap ){
    Edge orderedEdge = getOrderedEdge(edge);
    BlockSequence blockSeq = getBlockSequence();
    if (diffEdgeMatMap.count({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]}) == 0){
        diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, size_t>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.first]}, 0) );
    }
    diffEdgeMatMap[getOrderedPair<BlockIndex>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]})] += counter;
};

double StochasticBlockModelFamily::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) {
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgesInBlock = getEdgeCountsInBlock();
    vector<size_t> verticesInBLock = getVertexCountsInBlock();
    double logLikelihoodRatioTerm = 0;

    map<pair<BlockIndex, BlockIndex>, size_t> diffEdgeMatMap;
    map<BlockIndex, size_t> diffEdgesInBlockMap;

    for (auto edge : move.addedEdges){
        getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
    }
    for (auto edge : move.removedEdges){
        getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);
    }

    for (auto it=diffEdgeMatMap.begin(); it!= diffEdgeMatMap.end(); ++it){
        auto bu = it->first.first, bv = it->first.second;
        if (diffEdgesInBlockMap.count(bu) == 0) diffEdgesInBlockMap.insert(pair<BlockIndex, size_t>(bu, 0) );
        if (diffEdgesInBlockMap.count(bv) == 0) diffEdgesInBlockMap.insert(pair<BlockIndex, size_t>(bv, 0) );
        diffEdgesInBlockMap[bu] ++;
        diffEdgesInBlockMap[bv] ++;
        if (bu == bv){
            logLikelihoodRatioTerm += logDoubleFactorial(edgeMat[bu][bv] + 2 * it->second) - logDoubleFactorial(edgeMat[bu][bv]);
        }
        else{
            logLikelihoodRatioTerm += logFactorial(edgeMat[bu][bv] + it->second) - logFactorial(edgeMat[bu][bv]);
        }
    }

    for (auto it=diffEdgesInBlockMap.begin(); it!= diffEdgesInBlockMap.end(); ++it){
            auto r = it->first;
            logLikelihoodRatioTerm -= it->second *  log(verticesInBLock[r]);
    }
    return logLikelihoodRatioTerm;
};

void StochasticBlockModelFamily::getDiffAdjMatMapFromEdgeMove( const Edge& edge, int counter, map<pair<VertexIndex, VertexIndex>, size_t>& diffAdjMatMap){
    Edge orderedEdge = getOrderedEdge(edge);
    if (diffAdjMatMap.count({orderedEdge.first, orderedEdge.second}) == 0){
        diffAdjMatMap.insert( pair<pair<VertexIndex, VertexIndex>, size_t>({orderedEdge.first, orderedEdge.second}, 0) );
    }
    diffAdjMatMap[{orderedEdge.first, orderedEdge.second}] += counter;
};

double StochasticBlockModelFamily::getLogLikelihoodRatioAdjTerm (const GraphMove& move) {
    map<pair<VertexIndex, VertexIndex>, size_t> diffAdjMatMap;
    double logLikelihoodRatioTerm = 0;

    for (auto edge : move.addedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, 1, diffAdjMatMap);
    }
    for (auto edge : move.removedEdges){
        getDiffAdjMatMapFromEdgeMove(edge, -1, diffAdjMatMap);
    }
    for (auto it=diffAdjMatMap.begin(); it!= diffAdjMatMap.end(); ++it){
        auto u = it->first.first, v = it->first.second;
        auto edgeMult = m_state.getEdgeMultiplicityIdx(u, v);

        if (u == v){
            logLikelihoodRatioTerm -= logDoubleFactorial(2 * edgeMult + 2 * it->second) - logDoubleFactorial(2 * edgeMult);
        }
        else{
            logLikelihoodRatioTerm -= logFactorial(edgeMult + it->second) - logFactorial(edgeMult);
        }
    }

    return logLikelihoodRatioTerm;
};

double StochasticBlockModelFamily::getLogLikelihoodRatio (const GraphMove& move) {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

void StochasticBlockModelFamily::getDiffEdgeMatMapFromBlockMove( const BlockMove& move, map<pair<BlockIndex, BlockIndex>, size_t>& diffEdgeMatMap){
    BlockSequence blockSeq = getBlockSequence();
    for (auto neighbor : m_state.getNeighboursOfIdx(move.vertexIdx)){
        VertexIndex idx = neighbor.first;
        size_t edgeMult = neighbor.second;
        pair<BlockIndex, BlockIndex> orderedBlockPair = getOrderedPair<BlockIndex> ({move.prevBlockIdx, blockSeq[idx]});
        if (diffEdgeMatMap.count(getOrderedPair<BlockIndex>({orderedBlockPair.first, orderedBlockPair.second})) == 0){
            diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, size_t>(orderedBlockPair, 0));
        }
        diffEdgeMatMap[orderedBlockPair] --;

        orderedBlockPair = getOrderedPair<BlockIndex> ({move.nextBlockIdx, blockSeq[idx]});
        if (diffEdgeMatMap.count(getOrderedPair<BlockIndex>({orderedBlockPair.first, orderedBlockPair.second})) == 0){
            diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, size_t>(orderedBlockPair, 0));
        }
        diffEdgeMatMap[orderedBlockPair] ++;
    }
};

double StochasticBlockModelFamily::getLogLikelihoodRatio(const BlockMove& move){
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgesInBlock = getEdgeCountsInBlock();
    vector<size_t> verticesInBlock = getVertexCountsInBlock();
    double logLikelihoodRatio = 0;

    map<pair<BlockIndex, BlockIndex>, size_t> diffEdgeMatMap;
    map<BlockIndex, size_t> diffEdgesInBlockMap;

    getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);
    for (auto it=diffEdgeMatMap.begin(); it!= diffEdgeMatMap.end(); ++it){
        auto bu = it->first.first, bv = it->first.second;
        if (diffEdgesInBlockMap.count(bu) == 0) diffEdgesInBlockMap.insert(pair<BlockIndex, size_t>(bu, 0) );
        if (diffEdgesInBlockMap.count(bv) == 0) diffEdgesInBlockMap.insert(pair<BlockIndex, size_t>(bv, 0) );
        diffEdgesInBlockMap[bu] += it->second;
        diffEdgesInBlockMap[bv] += it->second;
        if (bu == bv){
            logLikelihoodRatio += logDoubleFactorial(edgeMat[bu][bv] + 2 * it->second) - logDoubleFactorial(edgeMat[bu][bv]);
        }
        else{
            logLikelihoodRatio += logFactorial(edgeMat[bu][bv] + it->second) - logFactorial(edgeMat[bu][bv]);
        }
    }

    for (auto it=diffEdgesInBlockMap.begin(); it!= diffEdgesInBlockMap.end(); ++it){
        if (it->first == move.nextBlockIdx && it->first < getBlockCount()){
            logLikelihoodRatio -= (edgesInBlock[ it->first ] + it->second) * log(verticesInBlock[ it->first ] + 1) ;
        }else if (it->first == move.prevBlockIdx){
            logLikelihoodRatio -= (edgesInBlock[ it->first ] + it->second) * log(verticesInBlock[ it->first - 1]) ;
        }
        if (it->first < getBlockCount()){
            logLikelihoodRatio += (edgesInBlock[ it->first ]) * log(verticesInBlock[ it->first ]) ;
        }
    }

    return logLikelihoodRatio;
};

EdgeMatrix StochasticBlockModelFamily::getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end());
    EdgeMatrix edgeMat(numBlocks, vector<size_t>(numBlocks, 0));

    size_t neighborIdx, edgeMult, r, s;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.first;
            edgeMult = neighbor.second;
            r = blockSeq[idx];
            s = blockSeq[neighborIdx];
            edgeMat[r][s] += edgeMult;
        }
    }

    return edgeMat;
};

void StochasticBlockModelFamily::checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat){
    EdgeMatrix actualEdgeMat = getEdgeMatrixFromGraph(graph, blockSeq);
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end());

    for (auto r = 0; r < numBlocks; r ++){
        for (auto s = 0; s < numBlocks; s ++){
            if (expectedEdgeMat[r][s] != actualEdgeMat[r][s])
                throw ConsistencyError("StochasticBlockModelFamily: edge matrix is inconsistent with graph.");
        }
    }
};

void StochasticBlockModelFamily::checkSelfConsistency(){
    m_blockPrior.checkSelfConsistency();
    m_edgeMatrixPrior.checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_state, getBlockSequence(), getEdgeMatrix());
}
