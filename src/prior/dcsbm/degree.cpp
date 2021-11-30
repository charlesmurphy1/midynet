#include <map>
#include <string>
#include <vector>

#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.h"

namespace FastMIDyNet{


void DegreePrior::setGraph(const MultiGraph& graph) {
    m_graph = &graph;
    const auto& blockSeq = getBlockSequence();
    const auto& blockCount = getBlockCount();

    m_state = graph.getDegrees();
    m_degreeCountsInBlocks = DegreeCountsMap(blockCount, 0);

    for (auto idx: graph){
        m_degreeCountsInBlocks[blockSeq[idx]].increment(m_state[idx]);
    }
}


void DegreePrior::setState(const DegreeSequence& degreeSeq) {
    m_degreeCountsInBlocks = computeDegreeCountsInBlocks(degreeSeq, getBlockSequence());
    m_state = degreeSeq;
    #if DEBUG
    checkSelfConsistency();
    #endif
}

std::vector<CounterMap<size_t>> DegreePrior::computeDegreeCountsInBlocks(const DegreeSequence& degreeSeq, const BlockSequence& blockSeq){
    if (blockSeq.size() != degreeSeq.size()) throw std::invalid_argument("blockSeq and degreeSeq have different sizes.");
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
    for (auto edge : move.addedEdges){
        if (edge.first != edge.second){
            m_degreeCountsInBlocks[blockSeq[edge.first]].decrement(degreeSeq[edge.first]);
            m_degreeCountsInBlocks[blockSeq[edge.second]].decrement(degreeSeq[edge.second]);
            m_degreeCountsInBlocks[blockSeq[edge.first]].increment(degreeSeq[edge.first] + 1);
            m_degreeCountsInBlocks[blockSeq[edge.second]].increment(degreeSeq[edge.second] + 1);
        } else {
            m_degreeCountsInBlocks[blockSeq[edge.first]].decrement(degreeSeq[edge.first]);
            m_degreeCountsInBlocks[blockSeq[edge.first]].increment(degreeSeq[edge.first] + 2);
        }
    }
    for (auto edge : move.removedEdges){
        if (edge.first != edge.second){
            m_degreeCountsInBlocks[blockSeq[edge.first]].decrement(degreeSeq[edge.first]);
            m_degreeCountsInBlocks[blockSeq[edge.second]].decrement(degreeSeq[edge.second]);
            m_degreeCountsInBlocks[blockSeq[edge.first]].increment(degreeSeq[edge.first] - 1);
            m_degreeCountsInBlocks[blockSeq[edge.second]].increment(degreeSeq[edge.second] - 1);
        } else {
            m_degreeCountsInBlocks[blockSeq[edge.first]].decrement(degreeSeq[edge.first]);
            m_degreeCountsInBlocks[blockSeq[edge.first]].increment(degreeSeq[edge.first] - 2);
        }
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
        applyGraphMoveToState(move); }
    );
    #if DEBUG
    checkSelfConsistency();
    #endif
}
void DegreePrior::applyBlockMove(const BlockMove& move) {
    processRecursiveFunction( [&]() {
        m_blockPrior.applyBlockMove(move);
        m_edgeMatrixPrior.applyBlockMove(move);
        applyBlockMoveToDegreeCounts(move);
        applyBlockMoveToState(move); }
    );
}

void DegreePrior::checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence& degreeSeq, size_t expectedEdgeCount){
    size_t actualEdgeCount = 0;
    for (auto k : degreeSeq) actualEdgeCount += k;
    if (actualEdgeCount != 2 * expectedEdgeCount)
        throw ConsistencyError("DegreePrior: degree sequence is inconsistent with expected edge count: "
        + std::to_string(actualEdgeCount) + "!=" + std::to_string(2 * expectedEdgeCount));
}

void DegreePrior::checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(
    const DegreeSequence& degreeSeq,
    const BlockSequence& blockSeq,
    const std::vector<CounterMap<size_t>>& expectedDegreeCountsInBlocks){
    if (degreeSeq.size() != blockSeq.size())
        throw std::invalid_argument("size of degreeSeq is inconsistent with size of blockSeq: "
        + std::to_string(degreeSeq.size()) + "!=" + std::to_string(blockSeq.size()));

    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    std::vector<CounterMap<size_t>> actualDegreeCountsInBlocks = DegreePrior::computeDegreeCountsInBlocks(degreeSeq, blockSeq);

    if (expectedDegreeCountsInBlocks.size() != actualDegreeCountsInBlocks.size())
        throw ConsistencyError("DegreePrior: expected degree counts are inconsistent with block count: "
        + std::to_string(expectedDegreeCountsInBlocks.size()) + "!=" + std::to_string(actualDegreeCountsInBlocks.size()));

    for (size_t r = 0; r < numBlocks; ++r){
        for (auto k : actualDegreeCountsInBlocks[r]){
            if ( expectedDegreeCountsInBlocks[r].isEmpty(k.first) )
                throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with degree sequence, n_"
                + std::to_string(k.first) + " is empty.");
            if ( expectedDegreeCountsInBlocks[r][k.first] != k.second )
                throw ConsistencyError("DegreePrior: expected degree counts is inconsistent with actual degree counts for n_"
                + std::to_string(k.first) + ": "
                + std::to_string(expectedDegreeCountsInBlocks[r][k.first]) + "!=" + std::to_string(k.second));
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
    std::vector<std::list<size_t>> degreeSeqInBlocks(getBlockCount());
    std::vector<std::list<size_t>::iterator> ptr_degreeSeqInBlocks(getBlockCount());
    const BlockSequence& blockSeq = getBlockSequence();
    const std::vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const std::vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();
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
    const std::vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks();
    const std::vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks();

    for (size_t r = 0; r < getBlockCount(); r++) {
        logLikelihood -= logMultisetCoefficient(vertexCountsInBlocks[r], edgeCountsInBlocks[r]);
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
        logLikelihoodRatio += logMultisetCoefficient(er + diff.second, nr) - logMultisetCoefficient(er, nr);
    }

    return logLikelihoodRatio;

}
double DegreeUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const{
    const BlockSequence& blockSeq = getBlockSequence();
    IntMap<BlockIndex> diffEdgeCountsInBlocksMap;
    IntMap<BlockIndex> diffVertexCountsInBlocksMap;
    diffVertexCountsInBlocksMap.decrement(move.prevBlockIdx);
    diffVertexCountsInBlocksMap.increment(move.nextBlockIdx);
    for (auto neighbor : m_graph->getNeighboursOfIdx(move.vertexIdx)){
        auto neighborIdx = neighbor.vertexIndex;
        auto neighborBlockIdx = blockSeq[neighborIdx];
        auto edgeMult = neighbor.label;
        diffEdgeCountsInBlocksMap.decrement(move.prevBlockIdx, edgeMult);
        diffEdgeCountsInBlocksMap.increment(move.nextBlockIdx, edgeMult);
    }

    const std::vector<size_t>& edgeCountsInBlocks = getEdgeCountsInBlocks() ;
    const std::vector<size_t>& vertexCountsInBlocks = getVertexCountsInBlocks() ;

    double logLikelihoodRatio = 0;
    for (auto diff : diffEdgeCountsInBlocksMap){
        auto er = edgeCountsInBlocks[diff.first];
        auto nr = vertexCountsInBlocks[diff.first];
        auto dnr = diffVertexCountsInBlocksMap.get(diff.first);
        logLikelihoodRatio += logMultisetCoefficient(er + diff.second, nr + dnr) - logMultisetCoefficient(er, nr);
    }

    return logLikelihoodRatio;
}

}// FastMIDyNet
