#include <map>
#include <random>
#include <string>
#include <tuple>
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
    m_degreeCountsInBlocks.clear();

    for (auto vertex: graph){
        m_degreeCountsInBlocks.increment({blockSeq[vertex], m_state[vertex]});
    }
}


void DegreePrior::setState(const DegreeSequence& degreeSeq) {
    m_degreeCountsInBlocks = computeDegreeCountsInBlocks(degreeSeq, m_blockPriorPtr->getState());
    m_state = degreeSeq;
}

CounterMap<std::pair<BlockIndex, size_t>> DegreePrior::computeDegreeCountsInBlocks(const DegreeSequence& degreeSeq, const BlockSequence& blockSeq){
    if (blockSeq.size() != degreeSeq.size()) throw invalid_argument("blockSeq and degreeSeq have different sizes.");
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    size_t maxDegree = *max_element(degreeSeq.begin(), degreeSeq.end()) + 1;
    DegreeCountsMap degreeCountsInBlocks;
    for (size_t i = 0; i < blockSeq.size(); i++) {
        degreeCountsInBlocks.increment({blockSeq[i], degreeSeq[i]});
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
        m_degreeCountsInBlocks.decrement({blockSeq[diff.first], degreeSeq[diff.first]});
        m_degreeCountsInBlocks.increment({blockSeq[diff.first], degreeSeq[diff.first] + diff.second});
    }
}

void DegreePrior::applyBlockMoveToDegreeCounts(const BlockMove& move){
    const DegreeSequence& degreeSeq = getState();
    if (move.nextBlockIdx == m_degreeCountsInBlocks.size()) onBlockCreation(move);
    m_degreeCountsInBlocks.decrement({move.prevBlockIdx, degreeSeq[move.vertexIdx]});
    m_degreeCountsInBlocks.increment({move.nextBlockIdx, degreeSeq[move.vertexIdx]});
}

void DegreePrior::_applyGraphMove(const GraphMove& move){
    m_blockPriorPtr->applyGraphMove(move);
    m_edgeMatrixPriorPtr->applyGraphMove(move);
    applyGraphMoveToDegreeCounts(move);
    applyGraphMoveToState(move);
}
void DegreePrior::_applyBlockMove(const BlockMove& move) {
    m_blockPriorPtr->applyBlockMove(move);
    m_edgeMatrixPriorPtr->applyBlockMove(move);
    applyBlockMoveToDegreeCounts(move);
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
    const DegreeCountsMap& expected){
    if (degreeSeq.size() != blockSeq.size())
        throw invalid_argument("size of degreeSeq is inconsistent with size of blockSeq: "
        + to_string(degreeSeq.size()) + "!=" + to_string(blockSeq.size()));

    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    DegreeCountsMap actual = DegreePrior::computeDegreeCountsInBlocks(degreeSeq, blockSeq);

    if (expected.size() != actual.size())
        throw ConsistencyError("DegreePrior: expected degree counts are inconsistent with actual degree counts: "
        + to_string(expected.size()) + "!=" + to_string(actual.size()));

    for (auto nk : actual){
        if ( expected.isEmpty(nk.first) )
            throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with degree sequence, since n_"
            + pairToString(nk.first) + " is empty while it registered " + std::to_string(nk.second) + ".");
        if ( expected.get(nk.first) != nk.second )
            throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with actual degree counts for n_"
            + pairToString(nk.first) + ": "
            + to_string(expected.get(nk.first)) + "!=" + to_string(nk.second));

    }
    // for (size_t r = 0; r < numBlocks; ++r){
    //     for (auto k : actualDegreeCountsInBlocks[r]){
    //     }
    // }
}

void DegreePrior::checkSelfConsistency() const{
    m_blockPriorPtr->checkConsistency();
    m_edgeMatrixPriorPtr->checkConsistency();
    checkDegreeSequenceConsistencyWithEdgeCount(getState(), m_edgeMatrixPriorPtr->getEdgeCount());
    checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(getState(), m_blockPriorPtr->getState(), getDegreeCountsInBlocks());
};

// void DegreePrior::onBlockCreation(const BlockMove& move){
//     m_degreeCountsInBlocks.push_back(CounterMap<size_t>());
// }

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
    const CounterMap<size_t>& edgeCountsInBlocks = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    const CounterMap<size_t>& vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();
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
    const CounterMap<size_t>& edgeCountsInBlocks = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    const CounterMap<size_t>& vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();

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

const double DegreeUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    BlockIndex r = move.prevBlockIdx, s = move.nextBlockIdx;
    size_t k = m_state[move.vertexIdx];
    size_t nr = m_blockPriorPtr->getVertexCountsInBlocks().get(r) ;
    size_t er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks().get(r) ;
    size_t eta_r = m_degreeCountsInBlocks.get({r, k});
    size_t ns = m_blockPriorPtr->getVertexCountsInBlocks().get(s) ;
    size_t es = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks().get(s) ;
    size_t eta_s =  m_degreeCountsInBlocks.get({s, k});

    double logLikelihoodRatio = 0;
    logLikelihoodRatio -= logMultisetCoefficient(er - k, nr - 1) - logMultisetCoefficient(er, nr);
    logLikelihoodRatio -= logMultisetCoefficient(es + k, ns + 1) - logMultisetCoefficient(es, ns);
    return logLikelihoodRatio;
}

void DegreeUniformHyperPrior::sampleState() {
    auto nr = m_blockPriorPtr->getVertexCountsInBlocks();
    auto er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    auto B = m_blockPriorPtr->getBlockCount();
    std::vector<std::list<size_t>> unorderedDegrees(B);
    for (size_t r = 0; r < B; ++r){
        auto p = sampleRandomRestrictedPartition(er[r], nr[r]);
        std::vector<size_t> v(p.begin(), p.end());
        std::shuffle(std::begin(v), std::end(v), rng);
        unorderedDegrees[r].assign(v.begin(), v.end());
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
    for (const auto& nk : m_degreeCountsInBlocks)
        logP += logFactorial(nk.second);
    for (const auto nr : m_blockPriorPtr->getVertexCountsInBlocks()){
        auto er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks().get(nr.first);
        if (er == 0)
            continue;
        logP -= logFactorial(nr.second);
        logP -= log_q_approx(er, nr.second);
    }
    return logP;
}
const double DegreeUniformHyperPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    IntMap<std::pair<BlockIndex, size_t>> diffDegreeMap;
    IntMap<BlockIndex> diffEdgeMap;
    for (auto edge : move.addedEdges){
        size_t ki = m_state[edge.first], kj = m_state[edge.second];
        BlockIndex r = m_blockPriorPtr->getBlockOfIdx(edge.first), s = m_blockPriorPtr->getBlockOfIdx(edge.second);
        diffDegreeMap.increment({r, ki + 1});
        diffDegreeMap.decrement({r, ki});
        diffDegreeMap.increment({s, kj + 1});
        diffDegreeMap.decrement({s, kj});

        diffEdgeMap.increment(m_blockPriorPtr->getBlockOfIdx(edge.first));
        diffEdgeMap.increment(m_blockPriorPtr->getBlockOfIdx(edge.second));
    }
    for (auto edge : move.removedEdges){
        size_t ki = m_state[edge.first], kj = m_state[edge.second];
        BlockIndex r = m_blockPriorPtr->getBlockOfIdx(edge.first), s = m_blockPriorPtr->getBlockOfIdx(edge.second);
        diffDegreeMap.increment({r, ki - 1});
        diffDegreeMap.decrement({r, ki});
        diffDegreeMap.increment({s, kj - 1});
        diffDegreeMap.decrement({s, kj});

        diffEdgeMap.decrement(m_blockPriorPtr->getBlockOfIdx(edge.first));
        diffEdgeMap.decrement(m_blockPriorPtr->getBlockOfIdx(edge.second));
    }

    double logLikelihoodRatio = 0;
    for (auto diff : diffDegreeMap){
        logLikelihoodRatio += logFactorial(m_degreeCountsInBlocks.get(diff.first) + diff.second);
        logLikelihoodRatio -= logFactorial(m_degreeCountsInBlocks.get(diff.first));
    }

    auto er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks();
    auto nr = m_blockPriorPtr->getVertexCountsInBlocks();

    for (auto diff : diffEdgeMap){
        logLikelihoodRatio -= log_q_approx(er[diff.first] + diff.second, nr[diff.first]);
        logLikelihoodRatio += log_q_approx(er[diff.first], nr[diff.first]);
    }

    return logLikelihoodRatio;

}

const double DegreeUniformHyperPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    BlockIndex r = move.prevBlockIdx, s = move.nextBlockIdx;
    bool createEmptyBlock = move.nextBlockIdx == m_blockPriorPtr->getVertexCountsInBlocks().size();
    size_t k = m_state[move.vertexIdx];
    size_t nr = m_blockPriorPtr->getVertexCountsInBlocks().get(r), ns = m_blockPriorPtr->getVertexCountsInBlocks().get(s) ;
    size_t er = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks().get(r), es = m_edgeMatrixPriorPtr->getEdgeCountsInBlocks().get(s) ;
    size_t eta_r = m_degreeCountsInBlocks.get({r, k}), eta_s = m_degreeCountsInBlocks.get({s, k});
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += log(eta_s + 1) - log(eta_r);
    logLikelihoodRatio -= log(ns + 1) - log(nr);
    logLikelihoodRatio -= log_q_approx(er - k, nr - 1) - log_q_approx(er, nr);
    logLikelihoodRatio -= log_q_approx(es + k, ns + 1);
    if (move.addedBlocks <= 0)
        logLikelihoodRatio -= -log_q_approx(es, ns);
    return logLikelihoodRatio;
}

}// FastMIDyNet
