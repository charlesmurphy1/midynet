#include <map>
#include <random>
#include <string>
#include <vector>

#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/integer_partition.h"
#include "FastMIDyNet/utility/maps.hpp"

using namespace std;

namespace FastMIDyNet{
void DegreePrior::setGraph(const MultiGraph& graph) {
    m_graph = &graph;
    m_edgeMatrixPriorPtr->setGraph(graph);
    const auto& blockSeq = m_blockPriorPtr->getState();
    const auto& blockCount = m_blockPriorPtr->getBlockCount();

    m_state = graph.getDegrees();
    m_degreeCountsInBlocks = vector<CounterMap<BlockIndex>>(blockCount, 0);

    for (auto vertex: graph){
        m_degreeCountsInBlocks[blockSeq[vertex]].increment(m_state[vertex]);
    }
}


void DegreePrior::setState(const DegreeSequence& degreeSeq) {
    m_degreeCountsInBlocks = computeDegreeCountsInBlocks(degreeSeq, m_blockPriorPtr->getState());
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
    const BlockSequence& blockSeq = m_blockPriorPtr->getState();

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
        m_blockPriorPtr->applyGraphMove(move);
        m_edgeMatrixPriorPtr->applyGraphMove(move);
        applyGraphMoveToDegreeCounts(move);
        applyGraphMoveToState(move);
    } );
}
void DegreePrior::applyBlockMove(const BlockMove& move) {
    processRecursiveFunction( [&]() {
        if (move.addedBlocks == 1) createBlock();
        m_blockPriorPtr->applyBlockMove(move);
        m_edgeMatrixPriorPtr->applyBlockMove(move);
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
    checkDegreeSequenceConsistencyWithEdgeCount(getState(), m_edgeMatrixPriorPtr->getEdgeCount());
    checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(getState(), m_blockPriorPtr->getState(), getDegreeCountsInBlocks());
};

void DegreePrior::createBlock(){
    m_degreeCountsInBlocks.push_back(0);
}

void DegreePrior::destroyBlock(const BlockIndex& blockIdx){
    m_degreeCountsInBlocks.erase(m_degreeCountsInBlocks.begin() + blockIdx);
}

const double DegreeDeltaPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    CounterMap<BaseGraph::VertexIndex> map;

    for (auto edge : move.addedEdges){
        map.increment(edge.first);
        map.increment(edge.second);
    }
    for (auto edge : move.addedEdges){
        map.decrement(edge.first);
        map.decrement(edge.second);
    }

    for (auto k: map){
        if (k.second != 0)
            return -INFINITY;
    }
    return 0.;
}

void DegreeUniformPrior::sampleState(){
    vector<list<size_t>> degreeSeqInBlocks(m_blockPriorPtr->getBlockCount());
    vector<list<size_t>::iterator> ptr_degreeSeqInBlocks(m_blockPriorPtr->getBlockCount());
    const BlockSequence& blockSeq = m_blockPriorPtr->getState();
    const vector<size_t>& edgeCountsInBlocks = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    const vector<size_t>& vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();
    for (size_t r = 0; r < m_blockPriorPtr->getBlockCount(); r++) {
        degreeSeqInBlocks[r] = sampleRandomWeakComposition(edgeCountsInBlocks[r], vertexCountsInBlocks[r]);
        ptr_degreeSeqInBlocks[r] = degreeSeqInBlocks[r].begin();
    }

    size_t size = m_blockPriorPtr->getSize();
    DegreeSequence degreeSeq(size);
    for (size_t i=0; i < size; ++i){
        auto r = blockSeq[i];
        degreeSeq[i] = *ptr_degreeSeqInBlocks[r];
        ++ptr_degreeSeqInBlocks[r];
    }

    setState(degreeSeq);
}

const double DegreeUniformPrior::getLogLikelihood() const{
    double logLikelihood = 0;
    const vector<size_t>& edgeCountsInBlocks = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    const vector<size_t>& vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();

    for (size_t r = 0; r < m_blockPriorPtr->getBlockCount(); r++) {
        logLikelihood -= logMultisetCoefficient(edgeCountsInBlocks[r], vertexCountsInBlocks[r]);
    }
    return logLikelihood;
}

const double DegreeUniformPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const{
    const BlockSequence& blockSeq = m_blockPriorPtr->getState();
    IntMap<BlockIndex> diffEdgeCountsInBlocksMap;
    for (auto edge : move.addedEdges){
        diffEdgeCountsInBlocksMap.increment(blockSeq[edge.first]) ;
        diffEdgeCountsInBlocksMap.increment(blockSeq[edge.second]) ;
    }

    for (auto edge : move.removedEdges){
        diffEdgeCountsInBlocksMap.decrement(blockSeq[edge.first]) ;
        diffEdgeCountsInBlocksMap.decrement(blockSeq[edge.second]) ;
    }

    const auto& edgeCountsInBlocks = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks() ;
    const auto& vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks() ;

    double logLikelihoodRatio = 0;
    for (auto diff : diffEdgeCountsInBlocksMap){
        auto er = edgeCountsInBlocks[diff.first];
        auto nr = vertexCountsInBlocks[diff.first];
        logLikelihoodRatio -= logMultisetCoefficient(er + diff.second, nr) - logMultisetCoefficient(er, nr);
    }

    return logLikelihoodRatio;

}
const double DegreeUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const{
    const BlockSequence& blockSeq = m_blockPriorPtr->getState();
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

    const vector<size_t>& edgeCountsInBlocks = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks() ;
    const vector<size_t>& vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks() ;
    double logLikelihoodRatio = 0;
    for (size_t r = 0 ; r < m_blockPriorPtr->getBlockCount() ; ++r){
        auto er = edgeCountsInBlocks[r];
        auto nr = vertexCountsInBlocks[r];
        auto dnr = diffVertexCountsInBlocksMap.get(r);
        auto der = diffEdgeCountsInBlocksMap.get(r);
        logLikelihoodRatio -= logMultisetCoefficient(er + der, nr + dnr) - logMultisetCoefficient(er, nr);
    }

    return logLikelihoodRatio;
}

void DegreeUniformHyperPrior::sampleState() {
    auto nr = m_blockPriorPtr->getVertexCountsInBlocks();
    auto er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    auto B = m_blockPriorPtr->getBlockCount();
    std::vector<std::list<size_t>> unorderedDegrees;
    for (size_t r = 0; r < B; ++r){
        unorderedDegrees[r] = sampleRandomRestrictedPartition(er[r], nr[r]);
        std::shuffle(std::begin(unorderedDegrees[r]), std::end(unorderedDegrees[r]), rng);
    }

    std::vector<size_t> degreeSeq(m_blockPriorPtr->getSize(), 0);
    for (size_t v=0; v<m_blockPriorPtr->getSize(); ++v){
        BlockIndex r = m_blockPriorPtr->getBlockOfIdx(v);
        degreeSeq[v] = unorderedDegrees[r].front();
        unorderedDegrees[r].pop_front();
    }
    setState(degreeSeq);
}

const double DegreeUniformHyperPrior::getLogLikelihood() const {
    double logP = 0;
    for (size_t r = 0; r < m_blockPriorPtr->getBlockCount(); ++r){
        auto nr = m_blockPriorPtr->getVertexCountsInBlocks()[r];
        auto er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks()[r];
        for (auto k : m_degreeCountsInBlocks[r].keys())
            logP += logFactorial(m_degreeCountsInBlocks[r].get(k));
        logP -= logFactorial(nr);
        logP -= logFactorial(log_q_approx(er, nr));
    }
    return logP;
}
const double DegreeUniformHyperPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    IntMap<BaseGraph::VertexIndex> diffDegreeMap;
    IntMap<BaseGraph::VertexIndex> diffEdgeMap;
    for (auto edge : move.addedEdges){
        diffDegreeMap.increment(edge.first);
        diffDegreeMap.increment(edge.second);

        diffEdgeMap.increment(m_blockPriorPtr->getBlockOfIdx(edge.first));
        diffEdgeMap.increment(m_blockPriorPtr->getBlockOfIdx(edge.second));
    }
    for (auto edge : move.removedEdges){
        diffDegreeMap.decrement(edge.first);
        diffDegreeMap.decrement(edge.second);

        diffEdgeMap.decrement(m_blockPriorPtr->getBlockOfIdx(edge.first));
        diffEdgeMap.decrement(m_blockPriorPtr->getBlockOfIdx(edge.second));
    }

    double logLikelihoodRatio = 0;
    for (auto diff : diffDegreeMap){
        size_t k = m_state[diff.first], r = m_blockPriorPtr->getBlockOfIdx(diff.first);
        logLikelihoodRatio += logFactorial(m_degreeCountsInBlocks[r].get(k) + diff.second);
        logLikelihoodRatio -= logFactorial(m_degreeCountsInBlocks[r].get(k));
    }

    for (auto diff : diffEdgeMap){
        logLikelihoodRatio -= logFactorial(m_edgeMatrixPriorPtr->getEdgeCountsInBlocks()[diff.first] + diff.second);
        logLikelihoodRatio += logFactorial(m_edgeMatrixPriorPtr->getEdgeCountsInBlocks()[diff.first]);
    }

    return logLikelihoodRatio;

}

const double DegreeUniformHyperPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    BlockIndex r = move.prevBlockIdx, s = move.nextBlockIdx;
    size_t k = m_state[move.vertexIdx];
    size_t nr = m_blockPriorPtr->getVertexCountsInBlocks()[r], ns = m_blockPriorPtr->getVertexCountsInBlocks()[s] ;
    size_t er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks()[r], es = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks()[s] ;
    
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += log(m_degreeCountsInBlocks[s].get(k) + 1) - log(m_degreeCountsInBlocks[r].get(k));
    logLikelihoodRatio -= logFactorial(er - k) + logFactorial(es + k) - logFactorial(er) - logFactorial(es);
    logLikelihoodRatio += log(nr) - log(ns + 1);
    return logLikelihoodRatio;
}

}// FastMIDyNet
