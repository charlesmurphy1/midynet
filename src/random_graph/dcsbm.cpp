#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/utility/functions.h"
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
};

void DegreeCorrectedStochasticBlockModelFamily::sampleState(){
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    DegreeSequence degreeSeq = getDegreeSequence();
    setState( generateDCSBM(blockSeq, edgeMat, degreeSeq) );
}


double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihood() const{
    double logLikelihood = 0;

    auto edgeMat = getEdgeMatrix() ;
    vector<size_t> edgesInBlock = getEdgeCountsInBlock();
    auto numBlocks = getBlockCount();
    for (size_t r = 0; r < numBlocks; r++) {
        logLikelihood += logDoubleFactorial(edgeMat[r][r]);
        logLikelihood -= logFactorial(edgesInBlock[r]);
        for (size_t s = r + 1; s < numBlocks; s++) {
            logLikelihood += logFactorial(edgeMat[r][s]);
        }
    }

    auto degreeSeq = getDegreeSequence();
    auto graph = getState();
    size_t neighborIdx, edgeMult;
    for (auto idx : graph){
        logLikelihood += logFactorial(degreeSeq[idx]);
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
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

double DegreeCorrectedStochasticBlockModelFamily::getLogPrior() const {
    return m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint() + m_degreePrior.getLogJoint();
};

double DegreeCorrectedStochasticBlockModelFamily::getLogJoint() const{
    return getLogLikelihood() + getLogPrior();
};

void DegreeCorrectedStochasticBlockModelFamily::getDiffEdgeMatMapFromEdgeMove( const Edge& edge, int counter, map<pair<BlockIndex, BlockIndex>, size_t>& diffEdgeMatMap ){
    Edge orderedEdge = getOrderedEdge(edge);
    BlockSequence blockSeq = getBlockSequence();
    if (diffEdgeMatMap.count({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]}) == 0){
        diffEdgeMatMap.insert( pair<pair<BlockIndex, BlockIndex>, size_t>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.first]}, 0) );
    }
    diffEdgeMatMap[getOrderedPair<BlockIndex>({blockSeq[orderedEdge.first], blockSeq[orderedEdge.second]})] += counter;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioEdgeTerm (const GraphMove& move) {
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgesInBlock = getEdgeCountsInBlock();
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
            auto bu = it->first;
            logLikelihoodRatioTerm -= logFactorial(edgesInBlock[bu] + it->second) - logFactorial(edgesInBlock[bu]);
    }
    return logLikelihoodRatioTerm;
};

void DegreeCorrectedStochasticBlockModelFamily::getDiffAdjMatMapFromEdgeMove( const Edge& edge, int counter, map<pair<VertexIndex, VertexIndex>, size_t>& diffAdjMatMap){
    Edge orderedEdge = getOrderedEdge(edge);
    if (diffAdjMatMap.count({orderedEdge.first, orderedEdge.second}) == 0){
        diffAdjMatMap.insert( pair<pair<VertexIndex, VertexIndex>, size_t>({orderedEdge.first, orderedEdge.second}, 0) );
    }
    diffAdjMatMap[{orderedEdge.first, orderedEdge.second}] += counter;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatioAdjTerm (const GraphMove& move) {
    DegreeSequence degreeSeq = getDegreeSequence();
    map<pair<VertexIndex, VertexIndex>, size_t> diffAdjMatMap;
    map<VertexIndex, size_t> diffDegreesMap;
    double logLikelihoodRatioTerm = 0;

    for (auto blockMove : move){
        for ( auto neighbor : m_state.getNeighboursOfIdx(blockMove.vertexIdx) ){
            neighborBlockIdx = blockSeq[neighborIdx];
            edgeMult = neighbor.label;
            nextEdgeMat[blockMove.prevBlockIdx][neighborBlockIdx] -= edgeMult;
            nextEdgeMat[neighborBlockIdx][blockMove.prevBlockIdx] -= edgeMult;
            nextEdgeMat[blockMove.nextBlockIdx][neighborBlockIdx] += edgeMult;
            nextEdgeMat[neighborBlockIdx][blockMove.nextBlockIdx] += edgeMult;
        }
        else{
            logLikelihoodRatioTerm -= logFactorial(edgeMult + it->second) - logFactorial(edgeMult);
        }
    }

    for (auto it=diffDegreesMap.begin(); it!= diffDegreesMap.end(); ++it){
            auto u = it->first;
            logLikelihoodRatioTerm += logFactorial(degreeSeq[u] + it->second) - logFactorial(degreeSeq[u]);
    }

    return logLikelihoodRatioTerm;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio (const GraphMove& move) {
    return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
}

void DegreeCorrectedStochasticBlockModelFamily::getDiffEdgeMatMapFromBlockMove( const BlockMove& move, map<pair<BlockIndex, BlockIndex>, size_t>& diffEdgeMatMap){
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

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio(const BlockMove& move){
    BlockSequence blockSeq = getBlockSequence();
    EdgeMatrix edgeMat = getEdgeMatrix();
    vector<size_t> edgesInBlock = getEdgeCountsInBlock();
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
            auto bu = it->first;
            logLikelihoodRatio -= logFactorial(edgesInBlock[bu] + it->second) - logFactorial(edgesInBlock[bu]);
    }
    return logLikelihoodRatio;
};

EdgeMatrix DegreeCorrectedStochasticBlockModelFamily::getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end());
    EdgeMatrix edgeMat(numBlocks, vector<size_t>(numBlocks, 0));

    size_t neighborIdx, edgeMult, r, s;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
            r = blockSeq[idx];
            s = blockSeq[neighborIdx];
            edgeMat[r][s] += edgeMult;
        }
    }

    return edgeMat;
};

DegreeSequence DegreeCorrectedStochasticBlockModelFamily::getDegreeSequenceFromGraph(const MultiGraph& graph){
    DegreeSequence degreeSeq(graph.getSize(), 0);

    size_t neighborIdx, edgeMult ;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.vertexIndex;
            edgeMult = neighbor.label;
            degreeSeq[idx] += edgeMult;
        }
    }

    return degreeSeq;
}

void DegreeCorrectedStochasticBlockModelFamily::checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat){
    EdgeMatrix actualEdgeMat = getEdgeMatrixFromGraph(graph, blockSeq);

    for (auto r = 0; r < actualEdgeMat.size(); r ++){
        for (auto s = 0; s < actualEdgeMat.size(); s ++){
            if (expectedEdgeMat[r][s] != actualEdgeMat[r][s])
                throw "Inconsistency error on edge matrix in DCSBM family.";
        }
    }
};

void DegreeCorrectedStochasticBlockModelFamily::checkGraphConsistencyWithDegreeSequence(const MultiGraph& graph, const DegreeSequence& expectedDegreeSeq){
    DegreeSequence actualDegreeSeq = getDegreeSequenceFromGraph(graph);

    for (auto idx : graph){
        if (expectedDegreeSeq[idx] != actualDegreeSeq[idx])
            throw "Inconsistency error on degree sequence in DCSBM family.";
    }

}

void DegreeCorrectedStochasticBlockModelFamily::checkSelfConsistency(){
    m_blockPrior.checkSelfConsistency();
    m_edgeMatrixPrior.checkSelfConsistency();
    m_degreePrior.checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_state, getBlockSequence(), getEdgeMatrix());
    checkGraphConsistencyWithDegreeSequence(m_state, getDegreeSequence());
}
