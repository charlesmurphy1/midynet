#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "FastMIDyNet/random_graph/prior/degree.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/integer_partition.h"
#include "FastMIDyNet/utility/maps.hpp"

using namespace std;

namespace FastMIDyNet{

const DegreeCountsMap DegreePrior::computeDegreeCounts(const std::vector<size_t>& degrees){
    DegreeCountsMap degreeCounts;
    for (size_t vertex=0; vertex<degrees.size(); ++vertex){
        degreeCounts.increment(degrees[vertex]);
    }
    return degreeCounts;
}

void DegreePrior::recomputeConsistentState() {
    m_degreeCounts = computeDegreeCounts(m_state);
}

void DegreePrior::setState(const DegreeSequence& state) {
    m_state = state;
    recomputeConsistentState();
}

void DegreePrior::setGraph(const MultiGraph& graph) {
    m_edgeCountPriorPtr->setState(graph.getTotalEdgeNumber());
    m_state = graph.getDegrees();
    recomputeConsistentState();
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
        m_degreeCounts.decrement(degreeSeq[diff.first]);
        m_degreeCounts.increment(degreeSeq[diff.first] + diff.second);
    }
}

void DegreePrior::_applyGraphMove(const GraphMove& move){
    m_edgeCountPriorPtr->applyGraphMove(move);
    applyGraphMoveToDegreeCounts(move);
    applyGraphMoveToState(move);
}

void DegreePrior::checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence& degreeSeq, size_t expectedEdgeCount){
    size_t actualEdgeCount = 0;
    for (auto k : degreeSeq) actualEdgeCount += k;
    if (actualEdgeCount != 2 * expectedEdgeCount)
        throw ConsistencyError("DegreePrior: degree sequence is inconsistent with expected edge count: "
        + to_string(actualEdgeCount) + "!=" + to_string(2 * expectedEdgeCount));
}

void DegreePrior::checkDegreeSequenceConsistencyWithDegreeCounts(
    const DegreeSequence& degreeSeq,
    const DegreeCountsMap& expected){
    DegreeCountsMap actual = DegreePrior::computeDegreeCounts(degreeSeq);
    if (expected.size() != actual.size())
        throw ConsistencyError("DegreePrior: expected degree counts are inconsistent with actual degree counts: "
        + to_string(expected.size()) + "!=" + to_string(actual.size()));

    for (auto nk : actual){
        if ( expected.isEmpty(nk.first) )
            throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with degree sequence, since n_"
            + std::to_string(nk.first) + " is empty while it registered " + std::to_string(nk.second) + ".");
        if ( expected.get(nk.first) != nk.second )
            throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with actual degree counts for n_"
            + std::to_string(nk.first) + ": "
            + std::to_string(expected.get(nk.first)) + "!=" + std::to_string(nk.second));

    }
}

void DegreePrior::checkSelfConsistency() const{
    m_edgeCountPriorPtr->checkConsistency();
    checkDegreeSequenceConsistencyWithEdgeCount(getState(), getEdgeCount());
    checkDegreeSequenceConsistencyWithDegreeCounts(getState(), getDegreeCounts());
};

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
    auto degreeList = sampleRandomWeakComposition(2 * getEdgeCount(), getSize());
    DegreeSequence degreeSeq;
    for (auto k : degreeList)
        degreeSeq.push_back(k);
    setState(degreeSeq);
}

void DegreeUniformHyperPrior::sampleState() {
    auto orderedDegreeList = sampleRandomRestrictedPartition(2 * getEdgeCount(), getSize());
    std::vector<size_t> degreeSeq(orderedDegreeList.begin(), orderedDegreeList.end());
    std::shuffle(std::begin(degreeSeq), std::end(degreeSeq), rng);
    setState(degreeSeq);
}

const double DegreeUniformHyperPrior::getLogLikelihood() const {
    double logLikelihood = -logFactorial(getSize()) - log_q_approx(getEdgeCount(), getSize());
    for (const auto& nk : m_degreeCounts)
        logLikelihood += logFactorial(nk.second);
    return logLikelihood;
}

const double DegreeUniformHyperPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    IntMap<size_t> diffDegreeMap;
    int dE = move.addedEdges.size() - move.removedEdges.size();

    for (auto edge : move.addedEdges){
        size_t ki = m_state[edge.first], kj = m_state[edge.second];
        diffDegreeMap.increment(ki + 1);
        diffDegreeMap.increment(kj + 1);
        diffDegreeMap.decrement(ki);
        diffDegreeMap.decrement(kj);
    }

    for (auto edge : move.removedEdges){
        size_t ki = m_state[edge.first], kj = m_state[edge.second];
        diffDegreeMap.increment(ki - 1);
        diffDegreeMap.increment(kj - 1);
        diffDegreeMap.decrement(ki);
        diffDegreeMap.decrement(kj);
    }

    double logLikelihoodRatio = log_q_approx(getEdgeCount(), getSize()) - log_q_approx(getEdgeCount() + dE, getSize());
    for (auto diff : diffDegreeMap){
        logLikelihoodRatio += logFactorial(m_degreeCounts.get(diff.first) + diff.second);
        logLikelihoodRatio -= logFactorial(m_degreeCounts.get(diff.first));
    }
    return logLikelihoodRatio;

}

}// FastMIDyNet
