#include <algorithm>
#include <tuple>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;

void DegreeCorrectedStochasticBlockModelFamily::sampleState(){
    m_blockPrior.sample();
    m_edgeMatrixPrior.sample();
    m_degreePrior.sample();
    BlockSequence blockSeq = m_blockPrior.getState();
    EdgeMatrix edgeMat = m_edgeMatrixPrior.getState();
    DegreeSequence degreeSeq = m_degreePrior.getState();

    setState( generateDCSBM(blockSeq, edgeMat, degreeSeq) );
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihood() const{
    double logLikelihood = 0;

    auto edgeMat = getEdgeMatrix() ;
    auto er = getEr(edgeMat);
    auto numBlocks = edgeMat.size();
    for (size_t r = 0; r < numBlocks; r++) {
        logLikelihood += logDoubleFactorial(edgeMat[r][r]);
        logLikelihood -= logFactorial(er[r]);
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

double DegreeCorrectedStochasticBlockModelFamily::getLogPrior() const {
    return m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint() + m_degreePrior.getLogJoint();
};

double DegreeCorrectedStochasticBlockModelFamily::getLogJoint() const{
    return getLogLikelihood() + getLogPrior();
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio (const std::vector<BaseGraph::Edge>& move, bool addition) const {
    double dS = 0;
    BlockSequence blockSeq = m_blockPrior.getState();
    EdgeMatrix edgeMat = m_edgeMatrixPrior.getState();
    DegreeSequence degreeSeq = m_degreePrior.getState();
    size_t numBlocks = edgeMat.size(), edgeMult;
    BaseGraph::VertexIndex u, v;
    vector<size_t> er = getEr(edgeMat);


    for (auto e : move){
        u = e.first, v = e.second;
        edgeMult = m_state.getEdgeMultiplicityIdx(u, v);
        if ( blockSeq[u] ==  blockSeq[v] )
        {
            dS += log(edgeMat[ blockSeq[u] ][ blockSeq[v] ] + 2.) - log(er[ blockSeq[u] ] + 1.) - log(er[ blockSeq[u] ] + 2.);
            if (u == v)
                dS += log(degreeSeq[u] + 1.) + log(degreeSeq[u] + 2.) - log(edgeMult + 2.);
            else
                dS += log(degreeSeq[u] + 1.) + log(degreeSeq[v] + 1.) - log(edgeMult + 1.);
        }
        else
            dS += log(edgeMat[ blockSeq[u] ][ blockSeq[v] ] + 1.) - log(er[ blockSeq[u] ] + 1.) - log(er[ blockSeq[v] ] + 1.) + log(degreeSeq[u] + 1.) + log(degreeSeq[v] + 1.) - log(edgeMult + 1.);
    }
    if (addition)
        return dS;
    else
        return -dS;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio(const vector<BlockMove>& move) const{
    double dS = 0;
    const BlockSequence& blockSeq = getBlockSequence();
    const EdgeMatrix& prevEdgeMat = getEdgeMatrix();
    EdgeMatrix nextEdgeMat = getEdgeMatrix();
    size_t numBlocks = prevEdgeMat.size(), edgeMult;
    BaseGraph::VertexIndex neighborIdx;
    BlockIndex neighborBlockIdx;
    vector<size_t> prevEr = getEr(prevEdgeMat), nextEr = getEr(prevEdgeMat);


    for (auto blockMove : move){
        for ( auto neighbor : m_state.getNeighboursOfIdx(blockMove.vertexIdx) ){
            neighborBlockIdx = blockSeq[neighborIdx];
            edgeMult = neighbor.second;
            nextEdgeMat[blockMove.prevBlockIdx][neighborBlockIdx] -= edgeMult;
            nextEdgeMat[neighborBlockIdx][blockMove.prevBlockIdx] -= edgeMult;
            nextEdgeMat[blockMove.nextBlockIdx][neighborBlockIdx] += edgeMult;
            nextEdgeMat[neighborBlockIdx][blockMove.nextBlockIdx] += edgeMult;
        }
        nextEr[blockMove.prevBlockIdx] -= edgeMult;
        nextEr[blockMove.nextBlockIdx] += edgeMult;
    }

    for (auto r = 0; r < numBlocks; r ++){
        dS += -logFactorial(nextEr[r]);
        dS -= -logFactorial(prevEr[r]);
        for (auto s = 0; s < numBlocks; s ++){
            if (s == r){
                dS += logDoubleFactorial(nextEdgeMat[r][s]);
                dS -= logDoubleFactorial(prevEdgeMat[r][s]);
            } else{
                dS += logFactorial(nextEdgeMat[r][s]);
                dS -= logFactorial(prevEdgeMat[r][s]);
            }
        }
    }
    return dS;
};


vector<size_t> DegreeCorrectedStochasticBlockModelFamily::getEr(const EdgeMatrix& edgeMat){
    size_t numBlocks = edgeMat.size();
    vector<size_t> er(numBlocks, 0);
    for (size_t r = 0; r < numBlocks; r++) {
        for (size_t s = 0; s < numBlocks; s++) {
            er[r] += edgeMat[r][s];
        }
    }
    return er;
};

EdgeMatrix DegreeCorrectedStochasticBlockModelFamily::getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
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

DegreeSequence DegreeCorrectedStochasticBlockModelFamily::getDegreeSequenceFromGraph(const MultiGraph& graph){
    DegreeSequence degreeSeq(graph.getSize(), 0);

    size_t neighborIdx, edgeMult ;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighborIdx = neighbor.first;
            edgeMult = neighbor.second;
            degreeSeq[idx] += edgeMult;
        }
    }

    return degreeSeq;
}

void DegreeCorrectedStochasticBlockModelFamily::checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat){
    EdgeMatrix actualEdgeMat = getEdgeMatrixFromGraph(graph, blockSeq);
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end());

    for (auto r = 0; r < numBlocks; r ++){
        for (auto s = 0; s < numBlocks; s ++){
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
