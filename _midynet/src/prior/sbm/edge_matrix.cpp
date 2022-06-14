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

    m_state = Matrix<size_t>(blockCount, std::vector<size_t>(blockCount, 0));
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
    size_t edgeCount = 0;
    for (auto vertex: graph) {
        const BlockIndex& r(blockSeq[vertex]);
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex > neighbor.vertexIndex)
                continue;

            const BlockIndex& s(blockSeq[neighbor.vertexIndex]);
            m_edgeCountsInBlocks[r]+=neighbor.label;
            m_edgeCountsInBlocks[s]+=neighbor.label;
            m_state[r][s]+=neighbor.label;
            m_state[s][r]+=neighbor.label;
            edgeCount += neighbor.label;
        }
    }
    m_edgeCountPriorPtr->setState(edgeCount);
}

void EdgeMatrixPrior::setState(const Matrix<size_t>& edgeMatrix) {
    m_state = edgeMatrix;

    const auto& blockCount = m_blockPriorPtr->getBlockCount();
    size_t edgeCount = 0;
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
    for (size_t i=0; i<blockCount; i++){
        for (size_t j=0; j<blockCount; j++){
            m_edgeCountsInBlocks[i] += edgeMatrix[i][j];
            edgeCount += edgeMatrix[i][j];
        }
    }
    m_edgeCountPriorPtr->setState(edgeCount);
}

void EdgeMatrixPrior::onBlockCreation(const BlockMove& move) {
    const auto& currentBlockCount = m_state.size();
    m_state.push_back(std::vector<size_t>(currentBlockCount, 0));
    m_edgeCountsInBlocks.push_back(0);
    for (auto& row: m_state)
        row.push_back(0);
}

void EdgeMatrixPrior::applyBlockMoveToState(const BlockMove& move) {
    if (move.prevBlockIdx == move.nextBlockIdx)
        return;

    const auto& blockSeq = m_blockPriorPtr->getState();
    const auto& degree = m_graphPtr->getDegreeOfIdx(move.vertexIdx);

    m_edgeCountsInBlocks[move.prevBlockIdx] -= degree;
    m_edgeCountsInBlocks[move.nextBlockIdx] += degree;
    for (auto neighbor: m_graphPtr->getNeighboursOfIdx(move.vertexIdx)) {
        auto neighborBlock = blockSeq[neighbor.vertexIndex];

        if (move.vertexIdx == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.prevBlockIdx;
        m_state[move.prevBlockIdx][neighborBlock] -= neighbor.label;
        m_state[neighborBlock][move.prevBlockIdx] -= neighbor.label;

        if (move.vertexIdx == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.nextBlockIdx;
        m_state[move.nextBlockIdx][neighborBlock] += neighbor.label;
        m_state[neighborBlock][move.nextBlockIdx] += neighbor.label;
    }
}

void EdgeMatrixPrior::applyGraphMoveToState(const GraphMove& move){
    const auto& blockSeq = m_blockPriorPtr->getState();

    for (auto removedEdge: move.removedEdges) {
        const BlockIndex& r(blockSeq[removedEdge.first]), s(blockSeq[removedEdge.second]);
        m_state[r][s]--;
        m_state[s][r]--;
        m_edgeCountsInBlocks[r] --;
        m_edgeCountsInBlocks[s] --;
    }
    for (auto addedEdge: move.addedEdges) {
        const BlockIndex& r(blockSeq[addedEdge.first]), s(blockSeq[addedEdge.second]);
        m_state[r][s]++;
        m_state[s][r]++;
        m_edgeCountsInBlocks[r] ++;
        m_edgeCountsInBlocks[s] ++;
    }
}

void EdgeMatrixPrior::checkSelfConsistency() const {
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeCountPriorPtr->checkSelfConsistency();

    const size_t& blockCount = m_blockPriorPtr->getBlockCount();
    verifyVectorHasAtLeastSize(m_edgeCountsInBlocks, blockCount, "EdgeMatrixPrior", "m_edgeCountInBlocks", "blockCount");
    verifyVectorHasAtLeastSize(m_state, blockCount, "EdgeMatrixPrior", "m_state", "blockCount");
    verifyVectorHasSize(m_state, m_edgeCountsInBlocks.size(), "EdgeMatrixPrior", "m_state", "m_edgeCountInBlocks");

    std::vector<size_t> actualEdgeCountsInBlocks(blockCount, 0);
    size_t sumEdges = 0;
    for (BlockIndex i=0; i<blockCount; i++) {
        size_t actualEdgeCountsInBlocks = 0;
        verifyVectorHasAtLeastSize(m_state[i], blockCount, "EdgeMatrixPrior", "m_state's row", "blocks");
        verifyVectorHasSize(m_state[i], m_edgeCountsInBlocks.size(), "EdgeMatrixPrior", "m_state's row", "m_edgeCountInBlocks");
        for (BlockIndex j=0; j<blockCount; j++) {
            if (m_state[i][j] != m_state[j][i])
                throw ConsistencyError("EdgeMatrixPrior: Edge matrix is not symmetric.");
            actualEdgeCountsInBlocks += m_state[i][j];
        }
        if (actualEdgeCountsInBlocks != m_edgeCountsInBlocks[i])
            throw ConsistencyError("EdgeMatrixPrior: Edge matrix row doesn't sum to edgeCountsInBlocks.");
        sumEdges += actualEdgeCountsInBlocks;
    }
    if (sumEdges != 2*m_edgeCountPriorPtr->getState())
        throw ConsistencyError("EdgeMatrixPrior: Sum of edge matrix isn't equal to twice the number of edges.");
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
    auto blockCount = m_blockPriorPtr->getBlockCount();
    auto flattenedEdgeMatrix = sampleRandomWeakComposition(
            m_edgeCountPriorPtr->getState(),
            blockCount*(blockCount+1)/2
            );
    m_state = EdgeMatrix(blockCount, std::vector<size_t>(blockCount, 0));
    std::pair<BlockIndex, BlockIndex> rs;
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
    size_t index = 0, correctedEdgeCount;
    for (auto edgeCountBetweenBlocks: flattenedEdgeMatrix) {
        rs = getUndirectedPairFromIndex(index, blockCount);
        m_edgeCountsInBlocks[rs.first] += edgeCountBetweenBlocks;
        m_edgeCountsInBlocks[rs.second] += edgeCountBetweenBlocks;
        m_state[rs.first][rs.second] += edgeCountBetweenBlocks;
        m_state[rs.second][rs.first] += edgeCountBetweenBlocks;
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

    bool creatingBlock = move.nextBlockIdx == m_blockPriorPtr->getBlockCount();
    bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx &&
                        vertexCountsInBlocks[move.prevBlockIdx] == 1;
    double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount() + creatingBlock - destroyingBlock, m_edgeCountPriorPtr->getState());
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
