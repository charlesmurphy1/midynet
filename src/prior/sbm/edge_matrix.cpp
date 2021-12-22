#include <string>
#include <vector>


#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"


namespace FastMIDyNet {


void EdgeMatrixPrior::setGraph(const MultiGraph& graph) {
    m_graph = &graph;
    const auto& blockSeq = m_blockPriorPtr->getState();
    const auto& blockCount = m_blockPriorPtr->getBlockCount();

    m_state = Matrix<size_t>(blockCount, std::vector<size_t>(blockCount, 0));
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);

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
        }
    }
}

void EdgeMatrixPrior::setState(const Matrix<size_t>& edgeMatrix) {
    m_state = edgeMatrix;

    const auto& blockCount = m_blockPriorPtr->getBlockCount();
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
    for (size_t i=0; i<blockCount; i++)
        for (size_t j=0; j<blockCount; j++)
            m_edgeCountsInBlocks[i] += edgeMatrix[i][j];
}

void EdgeMatrixPrior::createBlock() {
    const auto& currentBlockCount = m_state.size();
    m_state.push_back(std::vector<size_t>(currentBlockCount, 0));
    m_edgeCountsInBlocks.push_back(0);
    for (auto& row: m_state)
        row.push_back(0);
}

void EdgeMatrixPrior::destroyBlock(const BlockIndex& block) {
    m_state.erase(m_state.begin()+block);
    m_edgeCountsInBlocks.erase(m_edgeCountsInBlocks.begin()+block);

    for (auto& row: m_state)
        row.erase(row.begin()+block);
}

void EdgeMatrixPrior::moveEdgeCountsInBlocks(const BlockMove& move) {
    if (move.prevBlockIdx == move.nextBlockIdx)
        return;

    const auto& blockSeq = m_blockPriorPtr->getState();

    for (auto neighbor: m_graph->getNeighboursOfIdx(move.vertexIdx)) {
        if (neighbor.vertexIndex == move.vertexIdx) {
            m_state[move.prevBlockIdx][move.prevBlockIdx] -= 2*neighbor.label;
            m_edgeCountsInBlocks[move.prevBlockIdx] -= 2*neighbor.label;

            m_state[move.nextBlockIdx][move.nextBlockIdx] += 2*neighbor.label;
            m_edgeCountsInBlocks[move.nextBlockIdx] += 2*neighbor.label;
        }
        else {
            const BlockIndex& neighborBlock = blockSeq[neighbor.vertexIndex];
            m_state[move.prevBlockIdx][neighborBlock] -= neighbor.label;
            m_state[neighborBlock][move.prevBlockIdx] -= neighbor.label;
            m_edgeCountsInBlocks[move.prevBlockIdx] -= neighbor.label;

            m_state[move.nextBlockIdx][neighborBlock] += neighbor.label;
            m_state[neighborBlock][move.nextBlockIdx] += neighbor.label;
            m_edgeCountsInBlocks[move.nextBlockIdx] += neighbor.label;
        }
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



void EdgeMatrixPrior::applyBlockMoveToState(const BlockMove& move) {
    /* Must be computed before calling createBlock and destroyBlock because these methods
     * change m_edgeCountsInBlocks size*/

    if (move.addedBlocks == 1)
        createBlock();

    moveEdgeCountsInBlocks(move);

    if (move.addedBlocks == -1)
        destroyBlock(move.prevBlockIdx);
}

void EdgeMatrixPrior::checkSelfConsistency() const {
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeCountPriorPtr->checkSelfConsistency();

    const size_t& blockCount = m_blockPriorPtr->getBlockCount();
    verifyVectorHasSize(m_edgeCountsInBlocks, blockCount, "m_edgeCountInBlocks", "blocks");
    verifyVectorHasSize(m_state, blockCount, "Edge matrix", "blocks");

    std::vector<size_t> actualEdgeCountsInBlocks(blockCount, 0);
    size_t sumEdges = 0;
    for (BlockIndex i=0; i<blockCount; i++) {
        size_t actualEdgeCountsInBlocks = 0;
        verifyVectorHasSize(m_state[i], blockCount, "Edge matrix's row", "blocks");
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

double EdgeMatrixUniformPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState() + move.addedEdges.size() - move.removedEdges.size());
    return newLogLikelihood - currentLogLikelihood;
}

double EdgeMatrixUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    auto vertexCountsInBlocks = m_blockPriorPtr->getVertexCountsInBlocks();

    bool creatingBlock = move.nextBlockIdx == m_blockPriorPtr->getBlockCount();
    bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx &&
                        vertexCountsInBlocks[move.prevBlockIdx] == 1;
    double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount() + creatingBlock - destroyingBlock, m_edgeCountPriorPtr->getState());
    return newLogLikelihood - currentLogLikelihood;
}

} // namespace FastMIDyNet
