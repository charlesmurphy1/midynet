#include <map>
#include <string>
#include <vector>

#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.h"

using namespace std;

namespace FastMIDyNet{
void DegreePrior::setGraph(const MultiGraph& graph) {
    m_graph = &graph;
    m_edgeMatrixPrior.setGraph(graph);
    const auto& blockSeq = getBlockSequence();
    const auto& blockCount = getBlockCount();

    m_state = graph.getDegrees();
    m_degreeCountsInBlocks = vector<CounterMap<BlockIndex>>(blockCount, 0);

    for (auto vertex: graph){
        m_degreeCountsInBlocks[blockSeq[vertex]].increment(m_state[vertex]);
    }
}


void DegreePrior::setState(const DegreeSequence& degreeSeq) {
    m_degreeCountsInBlocks = computeDegreeCountsInBlocks(degreeSeq, getBlockSequence());
    m_state = degreeSeq;
}

vector<CounterMap<size_t>> DegreePrior::computeDegreeCountsInBlocks(const DegreeSequence& degreeSeq, const BlockSequence& blockSeq){
    if (blockSeq.size() != degreeSeq.size()) throw invalid_argument("blockSeq and degreeSeq have different sizes.");
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    size_t maxDegree = *max_element(degreeSeq.begin(), degreeSeq.end()) + 1;
    DegreeCountsMap degreeCountsInBlocks(numBlocks, 0);
    for (size_t i = 0; i < blockSeq.size(); i++) {
        degreeCountsInBlocks[blockSeq[i]].increment(degreeSeq[i]);
    }
    return degreeCountsInBlocks;
}

void DegreePrior::applyGraphMoveToState(const GraphMove& move){
    for (auto edge : move.addedEdges){
        ++m_state[edge.first];
        ++m_state[edge.second];
    }
    for (auto edge : move.removedEdges){
        --m_state[edge.first];
        --m_state[edge.second];
    }
}
void DegreePrior::applyGraphMoveToDegreeCounts(const GraphMove& move){
    const DegreeSequence& degreeSeq = getState();
    const BlockSequence& blockSeq = getBlockSequence();

    IntMap<BaseGraph::VertexIndex> diffDegreeMap;
    for (auto edge : move.addedEdges){
        diffDegreeMap.increment(edge.first);
        diffDegreeMap.increment(edge.second);
    }
    for (auto edge : move.removedEdges){
        diffDegreeMap.decrement(edge.first);
        diffDegreeMap.decrement(edge.second);
    }

    for (auto diff : diffDegreeMap){
        m_degreeCountsInBlocks[blockSeq[diff.first]].decrement(degreeSeq[diff.first]);
        m_degreeCountsInBlocks[blockSeq[diff.first]].increment(degreeSeq[diff.first] + diff.second);
    }
}

void DegreePrior::applyBlockMoveToDegreeCounts(const BlockMove& move){
    const DegreeSequence& degreeSeq = getState();
    m_degreeCountsInBlocks[move.prevBlockIdx].decrement(degreeSeq[move.vertexIdx]);
    m_degreeCountsInBlocks[move.nextBlockIdx].increment(degreeSeq[move.vertexIdx]);
}

void DegreePrior::applyGraphMove(const GraphMove& move){
    processRecursiveFunction( [&]() {
        m_blockPrior.applyGraphMove(move);
        m_edgeMatrixPrior.applyGraphMove(move);
        applyGraphMoveToDegreeCounts(move);
        applyGraphMoveToState(move);
    } );
}
void DegreePrior::applyBlockMove(const BlockMove& move) {
    processRecursiveFunction( [&]() {
        if (move.addedBlocks == 1) createBlock();
        m_blockPrior.applyBlockMove(move);
        m_edgeMatrixPrior.applyBlockMove(move);
        applyBlockMoveToDegreeCounts(move);
        applyBlockMoveToState(move);
        if (move.addedBlocks == -1) destroyBlock(move.prevBlockIdx);
    } );
}

void DegreePrior::checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence& degreeSeq, size_t expectedEdgeCount){
    size_t actualEdgeCount = 0;
    for (auto k : degreeSeq) actualEdgeCount += k;
    if (actualEdgeCount != 2 * expectedEdgeCount)
        throw ConsistencyError("DegreePrior: degree sequence is inconsistent with expected edge count: "
        + to_string(actualEdgeCount) + "!=" + to_string(2 * expectedEdgeCount));
}

void DegreePrior::checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(
    const DegreeSequence& degreeSeq,
    const BlockSequence& blockSeq,
    const vector<CounterMap<size_t>>& expectedDegreeCountsInBlocks){
    if (degreeSeq.size() != blockSeq.size())
        throw invalid_argument("size of degreeSeq is inconsistent with size of blockSeq: "
        + to_string(degreeSeq.size()) + "!=" + to_string(blockSeq.size()));

    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    vector<CounterMap<size_t>> actualDegreeCountsInBlocks = DegreePrior::computeDegreeCountsInBlocks(degreeSeq, blockSeq);

    if (expectedDegreeCountsInBlocks.size() != actualDegreeCountsInBlocks.size())
        throw ConsistencyError("DegreePrior: expected degree counts are inconsistent with block count: "
        + to_string(expectedDegreeCountsInBlocks.size()) + "!=" + to_string(actualDegreeCountsInBlocks.size()));

    for (size_t r = 0; r < numBlocks; ++r){
        for (auto k : actualDegreeCountsInBlocks[r]){
            if ( expectedDegreeCountsInBlocks[r].isEmpty(k.first) )
                throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with degree sequence, n_"
                + to_string(k.first) + " is empty but actually n_" + to_string(k.first)+" = "+to_string(k.second) + ".");
            if ( expectedDegreeCountsInBlocks[r][k.first] != k.second )
                throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with actual degree counts for n_"
                + to_string(k.first) + ": "
                + to_string(expectedDegreeCountsInBlocks[r][k.first]) + "!=" + to_string(k.second));
        }
    }
}

void DegreePrior::checkSelfConsistency() const{
    checkDegreeSequenceConsistencyWithEdgeCount(getState(), getEdgeCount());
    checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(getState(), getBlockSequence(), getDegreeCountsInBlocks());
};

void DegreePrior::createBlock(){
    m_degreeCountsInBlocks.push_back(0);
}

void DegreePrior::destroyBlock(const BlockIndex& blockIdx){
    m_degreeCountsInBlocks.erase(m_degreeCountsInBlocks.begin() + blockIdx);
}

void DegreeUniformPrior::sampleState(){
    vector<list<size_t>> degreeSeqInBlocks(getBlockCount());
    vector<list<size_t>::iterator> ptr_degreeSeqInBlocks(getBlockCount());
    const BlockSequence& blockSeq = getBlockSequence();
    const vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();
    for (size_t r = 0; r < getBlockCount(); r++) {
        degreeSeqInBlocks[r] = sampleRandomWeakComposition(edgeCountsInBlocks[r], vertexCountsInBlocks[r]);
        ptr_degreeSeqInBlocks[r] = degreeSeqInBlocks[r].begin();
    }

    DegreeSequence degreeSeq(getSize());
    for (size_t i=0; i < getSize(); ++i){
        auto r = blockSeq[i];
        degreeSeq[i] = *ptr_degreeSeqInBlocks[r];
        ++ptr_degreeSeqInBlocks[r];
    }

    setState(degreeSeq);
}

double DegreeUniformPrior::getLogLikelihood() const{
    double logLikelihood = 0;
    const vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();

    for (size_t r = 0; r < getBlockCount(); r++) {
        logLikelihood -= logMultisetCoefficient(edgeCountsInBlocks[r], vertexCountsInBlocks[r]);
    }
    return logLikelihood;
}

double DegreeUniformPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const{
    const BlockSequence& blockSeq = getBlockSequence();
    IntMap<BlockIndex> diffEdgeCountsInBlocksMap;
    for (auto edge : move.addedEdges){
        diffEdgeCountsInBlocksMap.increment(blockSeq[edge.first]) ;
        diffEdgeCountsInBlocksMap.increment(blockSeq[edge.second]) ;
    }

    for (auto edge : move.removedEdges){
        diffEdgeCountsInBlocksMap.decrement(blockSeq[edge.first]) ;
        diffEdgeCountsInBlocksMap.decrement(blockSeq[edge.second]) ;
    }

    const auto& edgeCountsInBlocks = getEdgeCountsInBlocks() ;
    const auto& vertexCountsInBlocks = getVertexCountsInBlocks() ;

    double logLikelihoodRatio = 0;
    for (auto diff : diffEdgeCountsInBlocksMap){
        auto er = edgeCountsInBlocks[diff.first];
        auto nr = vertexCountsInBlocks[diff.first];
        logLikelihoodRatio -= logMultisetCoefficient(er + diff.second, nr) - logMultisetCoefficient(er, nr);
    }

    return logLikelihoodRatio;

}
double DegreeUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const{
    const BlockSequence& blockSeq = getBlockSequence();
    IntMap<BlockIndex> diffEdgeCountsInBlocksMap;
    IntMap<BlockIndex> diffVertexCountsInBlocksMap;
    diffVertexCountsInBlocksMap.decrement(move.prevBlockIdx);
    diffVertexCountsInBlocksMap.increment(move.nextBlockIdx);
    for (auto neighbor : getGraph().getNeighboursOfIdx(move.vertexIdx)){
        auto neighborIdx = neighbor.vertexIndex;
        auto neighborBlockIdx = blockSeq[neighborIdx];
        auto edgeMult = neighbor.label;
        diffEdgeCountsInBlocksMap.decrement(move.prevBlockIdx, edgeMult);
        diffEdgeCountsInBlocksMap.increment(move.nextBlockIdx, edgeMult);
    }

    const vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks() ;
    const vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks() ;
    double logLikelihoodRatio = 0;
    for (size_t r = 0 ; r < getBlockCount() ; ++r){
        auto er = edgeCountsInBlocks[r];
        auto nr = vertexCountsInBlocks[r];
        auto dnr = diffVertexCountsInBlocksMap.get(r);
        auto der = diffEdgeCountsInBlocksMap.get(r);
        logLikelihoodRatio -= logMultisetCoefficient(er + der, nr + dnr) - logMultisetCoefficient(er, nr);
    }

    return logLikelihoodRatio;
}

}// FastMIDyNet
