#include <algorithm>
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
    BLockSequence blockSeq = m_blockPrior.getState();
    EdgeMatrix edgeMat = m_edgeMatrixPrior.getState();
    DegreeSequence degreeSeq = m_degreePrior.getState();

    return generateDCSBM(blockSeq, edgeMat, degreeSeq);
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihood(){
    double logLikelihood = 0;

    auto edgeMat = getEdgeMatrix();
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
    size_t neighbor_idx, edgeMultiplicity;
    for (auto idx : graph){
        logLikelihood += logFactorial(degreeSeq[idx]);
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighbor_idx = neighbor.first;
            edgeMultiplicity = neighbor.second;
            if (idx < neighbor_idx){
                continue
            }else if (idx == neighbor_idx){
                logLikelihood -= logDoubleFactorial(edgeMultiplicity);
            }else{
                logLikelihood -= logFactorial(edgeMultiplicity);
            }
        }
    }

    return logLikelihood;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogPrior(){
    return m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint() + m_degreePrior.getLogJoint();
};

double DegreeCorrectedStochasticBlockModelFamily::getLogJoint(){
    return getLogLikelihood() + getLogPrior();
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio(const EdgeMove& move, bool addition){
    double dS = 0;
    BlockSequence blockSeq = m_blockPrior.getState();
    EdgeMatrix edgeMat = m_edgeMatrixPrior.getState();
    DegreeSequence degreeSeq = m_degreePrior.getState();
    size_t numBlocks = edgeMat.size();
    BaseGraph::VertexIndex u, v;
    vector<size_t> er = getEr(edgeMat);


    for (auto e : move){
        u = e.first, v = e.second;
        edgeMultiplicity = m_state.getEdgeMultiplicityIdx(u, v);
        if ( blockSeq[u] ==  blockSeq[v] )
        {
            dS += log(edgeMat[ blockSeq[u] ][ blockSeq[v] ] + 2.) - log(er[ blockSeq[u] ] + 1.) - log(er[ blockSeq[u] ] + 2.);
            if (u == v)
                dS += log(degreeSeq[u] + 1.) + log(degreeSeq[u] + 2.) - log(m_state.getEdgeMultiplicityIdx(u, v); + 2.);
            else
                dS += log(degreeSeq[u] + 1.) + log(degreeSeq[v] + 1.) - log(m_state.getEdgeMultiplicityIdx(u, v); + 1.);
        }
        else
            dS += log(edgeMat[ blockSeq[u] ][ blockSeq[v] ] + 1.) - log(er[ blockSeq[u] ] + 1.) - log(er[ blockSeq[v] ] + 1.) + log(degreeSeq[u] + 1.) + log(degreeSeq[v] + 1.) - log(m_state.getEdgeMultiplicityIdx(u, v); + 1.);
    }
    if (addition)
        return dS;
    else
        return -dS;
};

double DegreeCorrectedStochasticBlockModelFamily::getLogLikelihoodRatio(const BlockMove& move){
    double dS = 0;
    BlockSequence blockSeq = m_blockPrior.getState();
    EdgeMatrix edgeMat = m_edgeMatrixPrior.getState();
    DegreeSequence degreeSeq = m_degreePrior.getState();
    size_t numBlocks = edgeMat.size();
    BaseGraph::VertexIndex u, v;
    vector<size_t> er = getEr(edgeMat);


    for (auto e : move){
        u = e.first, v = e.second;
        edgeMultiplicity = m_state.getEdgeMultiplicityIdx(u, v);
        if ( blockSeq[u] ==  blockSeq[v] )
        {
            dS += log(edgeMat[ blockSeq[u] ][ blockSeq[v] ] + 2.) - log(er[ blockSeq[u] ] + 1.) - log(er[ blockSeq[u] ] + 2.);
            if (u == v)
                dS += log(degreeSeq[u] + 1.) + log(degreeSeq[u] + 2.) - log(m_state.getEdgeMultiplicityIdx(u, v); + 2.);
            else
                dS += log(degreeSeq[u] + 1.) + log(degreeSeq[v] + 1.) - log(m_state.getEdgeMultiplicityIdx(u, v); + 1.);
        }
        else
            dS += log(edgeMat[ blockSeq[u] ][ blockSeq[v] ] + 1.) - log(er[ blockSeq[u] ] + 1.) - log(er[ blockSeq[v] ] + 1.) + log(degreeSeq[u] + 1.) + log(degreeSeq[v] + 1.) - log(m_state.getEdgeMultiplicityIdx(u, v); + 1.);
    }
    if (addition)
        return dS;
    else
        return -dS;
};


static double DegreeCorrectedStochasticBlockModelFamily::getEr(const EdgeMatrix& edgeMat) const{
    size_t numBlocks = edgeMat.size();
    vector<size_t> er(numBLocks, 0);
    for (size_t r = 0; r < numBLocks; r++) {
        for (size_t s = 0; s < numBLocks; s++) {
            er[r] += edgeMat[r][s];
        }
    }
    return er;
};

void DegreeCorrectedStochasticBlockModelFamily::checkGraphConsistency(const MultiGraph& graph, BlockSequence blockSeq, EdgeMatrix edgeMat, DegreeSequence degreeSeq){
    size_t numBlocks = getBlockCount();
    EdgeMatrix expect_edgeMat(numBlocks, vector<size_t>(numBlocks, 0));
    DegreeSequence expect_degreeSeq( getSize() );

    size_t neighbor_idx, edgeMultiplicity, r, s;
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            neighbor_idx = neighbor.first;
            edgeMultiplicity = neighbor.second;
            r = blockSeq[idx];
            s = blockSeq[neighbor_idx];
            expect_edgeMat[r][s] += edgeMultiplicity;
            expect_degreeSeq[idx] += edgeMultiplicity;
        }
        if (expect_degreeSeq[idx] != degreeSeq[idx])
            throw "Inconsistency error on degree sequence.";
    }

    for (auto r_idx = 0; r_idx < numBlocks; r_idx ++){
        for (auto s_idx = 0; s_idx < numBlocks; s_idx ++){
            if (expect_edgeMat[r][s] != edgeMat[r][s])
                throw "Inconsistency error on edge matrix in DCSBM family.";
        }
    }
};
