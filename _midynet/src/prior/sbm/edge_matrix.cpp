#include <string>
#include <vector>


#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"


namespace FastMIDyNet {

/* DEFINITION OF EDGE MATRIX PRIOR BASE CLASS */

void EdgeMatrixPrior::setGraph(const MultiGraph& graph) {
    m_graphPtr = &graph;
    const auto& blockSeq = m_blockPriorPtr->getState();
    const auto& blockCount = m_blockPriorPtr->getBlockCount();

    m_state = MultiGraph(blockCount);
    m_edgeCountsInBlocks.clear();
    size_t edgeCount = 0;
    for (auto vertex: graph) {
        const BlockIndex& r(blockSeq[vertex]);
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex > neighbor.vertexIndex)
                continue;

            const BlockIndex& s(blockSeq[neighbor.vertexIndex]);
            m_edgeCountsInBlocks.increment(r, neighbor.label);
            m_edgeCountsInBlocks.increment(s, neighbor.label);
            m_state.addMultiedgeIdx(r, s, neighbor.label);
        }
    }
    m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
}

void EdgeMatrixPrior::setState(const MultiGraph& edgeMatrix) {
    m_state = edgeMatrix;
    m_edgeCountsInBlocks.clear();
    for (auto r : m_state)
        m_edgeCountsInBlocks.set(r, m_state.getDegreeOfIdx(r));
    m_edgeCountPriorPtr->setState(edgeMatrix.getTotalEdgeNumber());
}

void EdgeMatrixPrior::applyBlockMoveToState(const BlockMove& move) {
    if (move.prevBlockIdx == move.nextBlockIdx)
        return;

    if (m_state.getSize() == move.nextBlockIdx)
        m_state.resize(move.nextBlockIdx + 1);
    const auto& blockSeq = m_blockPriorPtr->getState();
    const auto& degree = m_graphPtr->getDegreeOfIdx(move.vertexIdx);

    m_edgeCountsInBlocks.decrement(move.prevBlockIdx, degree);
    m_edgeCountsInBlocks.increment(move.nextBlockIdx, degree);
    for (auto neighbor: m_graphPtr->getNeighboursOfIdx(move.vertexIdx)) {
        auto neighborBlock = blockSeq[neighbor.vertexIndex];

        if (move.vertexIdx == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.prevBlockIdx;
        m_state.removeMultiedgeIdx(move.prevBlockIdx, neighborBlock, neighbor.label) ;

        if (move.vertexIdx == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.nextBlockIdx;
        m_state.addMultiedgeIdx(move.nextBlockIdx, neighborBlock, neighbor.label) ;
    }

}

void EdgeMatrixPrior::applyGraphMoveToState(const GraphMove& move){
    const auto& blockSeq = m_blockPriorPtr->getState();

    for (auto removedEdge: move.removedEdges) {
        const BlockIndex& r(blockSeq[removedEdge.first]), s(blockSeq[removedEdge.second]);
        m_state.removeEdgeIdx(r, s);
        m_edgeCountsInBlocks.decrement(r);
        m_edgeCountsInBlocks.decrement(s);
    }
    for (auto addedEdge: move.addedEdges) {
        const BlockIndex& r(blockSeq[addedEdge.first]), s(blockSeq[addedEdge.second]);
        m_state.addEdgeIdx(r, s);
        m_edgeCountsInBlocks.increment(r);
        m_edgeCountsInBlocks.increment(s);
    }
}

void EdgeMatrixPrior::checkSelfConsistency() const {
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeCountPriorPtr->checkSelfConsistency();

    const size_t& blockCount = m_blockPriorPtr->getBlockCount();
    size_t sumEdges = 0;
    for (BlockIndex r : m_state) {
        size_t actualEdgeCountsInBlocks = m_state.getDegreeOfIdx(r);
        if (actualEdgeCountsInBlocks != m_edgeCountsInBlocks[r])
            throw ConsistencyError("EdgeMatrixPrior: Edge matrix row ("
            + std::to_string(actualEdgeCountsInBlocks) + ") doesn't sum to edgeCountsInBlocks["
            + std::to_string(r) + "] (" + std::to_string(m_edgeCountsInBlocks[r]) + ").");
        sumEdges += actualEdgeCountsInBlocks;
    }
    if (sumEdges != 2*m_edgeCountPriorPtr->getState())
        throw ConsistencyError("EdgeMatrixPrior: Sum of edge matrix isn't equal to the number of edges.");
}


const double EdgeMatrixDeltaPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    CounterMap<std::pair<BlockIndex, BlockIndex>> map;

    for (auto edge : move.addedEdges){
        BlockIndex r = m_blockPriorPtr->getBlockOfIdx(edge.first);
        BlockIndex s = m_blockPriorPtr->getBlockOfIdx(edge.second);
        map.decrement({r, s});
        map.decrement({s, r});
    }
    for (auto edge : move.addedEdges){
        BlockIndex r = m_blockPriorPtr->getBlockOfIdx(edge.first);
        BlockIndex s = m_blockPriorPtr->getBlockOfIdx(edge.second);
        map.increment({r, s});
        map.increment({s, r});
    }

    for (auto k: map){
        if (k.second != 0)
            return -INFINITY;
    }
    return 0.;
}

const double EdgeMatrixDeltaPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    if (move.prevBlockIdx != move.nextBlockIdx)
        return -INFINITY;
    return 0.;
}



/* DEFINITION OF EDGE MATRIX UNIFORM PRIOR */

void EdgeMatrixUniformPrior::sampleState() {

    const auto& blockCount = m_blockPriorPtr->getBlockCount();
    const auto& blocks = m_blockPriorPtr->getState();
    auto flattenedEdgeMatrix = sampleRandomWeakComposition(
            m_edgeCountPriorPtr->getState(),
            blockCount*(blockCount+1)/2
            );
    m_state = MultiGraph(*max_element(blocks.begin(), blocks.end()) + 1);
    std::pair<BlockIndex, BlockIndex> rs;
    m_edgeCountsInBlocks.clear();
    size_t index = 0, correctedEdgeCount;
    for (auto edgeCountBetweenBlocks: flattenedEdgeMatrix) {
        rs = getUndirectedPairFromIndex(index, blockCount);
        m_edgeCountsInBlocks.increment(rs.first, edgeCountBetweenBlocks);
        m_edgeCountsInBlocks.increment(rs.second, edgeCountBetweenBlocks);
        m_state.addMultiedgeIdx(rs.first, rs.second, edgeCountBetweenBlocks);
        index++;
    }
}

const double EdgeMatrixUniformPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState() + move.addedEdges.size() - move.removedEdges.size());
    return newLogLikelihood - currentLogLikelihood;
}

const double EdgeMatrixUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    auto vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();

    // bool creatingBlock = move.nextBlockIdx == m_blockPriorPtr->getBlockCount();
    // bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx && vertexCountsInBlocks[move.prevBlockIdx] == 1;
    int addedBlocks = m_blockPriorPtr->getAddedBlocks(move);
    double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount() + addedBlocks, m_edgeCountPriorPtr->getState());
    return newLogLikelihood - currentLogLikelihood;
}

// /* DEFINITION OF EDGE MATRIX EXPONENTIAL PRIOR */
//
// void EdgeMatrixExponentialPrior::sampleState() {
//     auto blockCount = m_blockPriorPtr->getBlockCount();
//     auto flattenedEdgeMatrix = sampleRandomWeakComposition(
//             m_edgeCountPriorPtr->getState(),
//             blockCount*(blockCount+1)/2
//             );
//
//     m_state = EdgeMatrix(blockCount, std::vector<size_t>(blockCount, 0));
//     std::pair<BlockIndex, BlockIndex> rs;
//     m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
//     size_t index = 0, correctedEdgeCount;
//     for (auto edgeCountBetweenBlocks: flattenedEdgeMatrix) {
//         rs = getUndirectedPairFromIndex(index, blockCount);
//         m_edgeCountsInBlocks[rs.first] += edgeCountBetweenBlocks;
//         m_edgeCountsInBlocks[rs.second] += edgeCountBetweenBlocks;
//         m_state[rs.first][rs.second] += edgeCountBetweenBlocks;
//         m_state[rs.second][rs.first] += edgeCountBetweenBlocks;
//         index++;
//     }
// }
//
// double EdgeMatrixExponentialPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
//     double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
//     double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState() + move.addedEdges.size() - move.removedEdges.size());
//     return newLogLikelihood - currentLogLikelihood;
// }
//
// double EdgeMatrixExponentialPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
//     auto vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();
//
//     bool creatingBlock = move.nextBlockIdx == m_blockPriorPtr->getBlockCount();
//     bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx &&
//                         vertexCountsInBlocks[move.prevBlockIdx] == 1;
//     double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
//     double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount() + creatingBlock - destroyingBlock, m_edgeCountPriorPtr->getState());
//     return newLogLikelihood - currentLogLikelihood;
// }



} // namespace FastMIDyNet
