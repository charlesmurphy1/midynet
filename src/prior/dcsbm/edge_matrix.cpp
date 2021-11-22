#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include <string>


namespace FastMIDyNet {


void EdgeMatrixPrior::setGraph(const MultiGraph& graph) {
    m_graph = &graph;
    const auto& vertexBlocks = m_blockPrior.getState();
    const auto& blockCount = m_blockPrior.getBlockCount();

    m_state = Matrix<size_t>(blockCount, std::vector<size_t>(blockCount, 0));
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);

    for (auto vertex: graph) {
        const BlockIndex& r(vertexBlocks[vertex]);
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex > neighbor.first)
                continue;

            const BlockIndex& s(vertexBlocks[neighbor.first]);
            m_edgeCountsInBlocks[r]+=neighbor.second;
            m_edgeCountsInBlocks[s]+=neighbor.second;
            m_state[r][s]+=neighbor.second;
            m_state[s][r]+=neighbor.second;
        }
    }
}

void EdgeMatrixPrior::setState(const Matrix<size_t>& edgeMatrix) {
    m_state = edgeMatrix;

    const auto& blockCount = m_blockPrior.getBlockCount();
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
    for (size_t i=0; i<blockCount; i++)
        for (size_t j=0; j<blockCount; j++)
            m_edgeCountsInBlocks[i] += edgeMatrix[i][j];
}

void EdgeMatrixPrior::createBlock() {
    const auto& blockCount = m_blockPrior.getBlockCount();

    m_state.push_back(std::vector<size_t>(blockCount, 0));
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

    const auto& vertexBlocks = m_blockPrior.getState();

    for (auto neighbor: m_graph->getNeighboursOfIdx(move.vertexIdx)) {
        if (neighbor.first == move.vertexIdx) {
            m_state[move.prevBlockIdx][move.prevBlockIdx] -= 2*neighbor.second;
            m_edgeCountsInBlocks[move.prevBlockIdx] -= 2*neighbor.second;

            m_state[move.nextBlockIdx][move.nextBlockIdx] += 2*neighbor.second;
            m_edgeCountsInBlocks[move.nextBlockIdx] += 2*neighbor.second;
        }
        else {
            const BlockIndex& neighborBlock = vertexBlocks[neighbor.first];
            m_state[move.prevBlockIdx][neighborBlock] -= neighbor.second;
            m_state[neighborBlock][move.prevBlockIdx] -= neighbor.second;
            m_edgeCountsInBlocks[move.prevBlockIdx] -= neighbor.second;

            m_state[move.nextBlockIdx][neighborBlock] += neighbor.second;
            m_state[neighborBlock][move.nextBlockIdx] += neighbor.second;
            m_edgeCountsInBlocks[move.nextBlockIdx] += neighbor.second;
        }
    }
}

void EdgeMatrixPrior::applyMoveToState(const GraphMove& move){
    const auto& vertexBlocks = m_blockPrior.getState();

    for (auto addedEdge: move.addedEdges) {
        const BlockIndex& r(vertexBlocks[addedEdge.first]), s(vertexBlocks[addedEdge.second]);
        m_state[r][s]++;
        m_state[s][r]++;
        m_edgeCountsInBlocks[r]++;
        m_edgeCountsInBlocks[s]++;
    }
    for (auto removedEdge: move.removedEdges) {
        const BlockIndex& r(vertexBlocks[removedEdge.first]), s(vertexBlocks[removedEdge.second]);
        m_state[r][s]--;
        m_state[s][r]--;
        m_edgeCountsInBlocks[r]--;
        m_edgeCountsInBlocks[s]--;
    }
}



void EdgeMatrixPrior::applyMoveToState(const BlockMove& move) {
    /* Must be computed before calling createBlock and destroyBlock because these methods
     * change m_edgeCountsInBlocks size*/
    bool creatingBlock = m_edgeCountsInBlocks.size()+1 == m_blockPrior.getBlockCount();
    bool destroyingBlock = m_edgeCountsInBlocks.size() == m_blockPrior.getBlockCount()+1;

    if (creatingBlock)
        createBlock();

    moveEdgeCountsInBlocks(move);

    if (destroyingBlock)
        destroyBlock(move.prevBlockIdx);
}

template<typename T>
static void verifyVectorHasSize(
    const std::vector<T>& vec,
    size_t size,
    const std::string& vectorName,
    const std::string& sizeName) {
    if (vec.size() != size)
        throw ConsistencyError("EdgeMatrixPrior: "+vectorName+" has size "+
                std::to_string(vec.size())+" while there are "+
                std::to_string(size)+" "+sizeName+".");
}

void EdgeMatrixPrior::checkSelfConsistency() const {
    m_blockPrior.checkSelfConsistency();
    m_edgeCountPrior.checkSelfConsistency();

    const size_t& blockCount = m_blockPrior.getBlockCount();
    verifyVectorHasSize(m_edgeCountsInBlocks, blockCount, "m_edgeCoutInBlocks", "blocks");
    verifyVectorHasSize(m_state, blockCount, "Edge matrix", "blocks");

    std::vector<size_t> actualEdgesInBlock(blockCount, 0);
    size_t sumEdges = 0;
    for (BlockIndex i=0; i<blockCount; i++) {
        size_t actualEdgesInBlock = 0;
        verifyVectorHasSize(m_state[i], blockCount, "Edge matrix's row", "blocks");

        for (BlockIndex j=0; j<blockCount; j++) {
            if (m_state[i][j] != m_state[j][i])
                throw ConsistencyError("EdgeMatrixPrior: Edge matrix is not symmetric.");
            actualEdgesInBlock += m_state[i][j];
        }
        if (actualEdgesInBlock != m_edgeCountsInBlocks[i])
            throw ConsistencyError("EdgeMatrixPrior: Edge matrix row doesn't sum to edgesInBlock.");
        sumEdges += actualEdgesInBlock;
    }
    if (sumEdges != 2*m_edgeCountPrior.getState())
        throw ConsistencyError("EdgeMatrixPrior: Sum of edge matrix isn't equal to twice the number of edges.");
}


double EdgeMatrixDeltaPrior::getLogLikelihoodRatio(const GraphMove& move) const{
    return -INFINITY;
}

double EdgeMatrixDeltaPrior::getLogLikelihoodRatio(const BlockMove& move) const{
    return -INFINITY;
}

void EdgeMatrixUniformPrior::sampleState() {
    auto blockCount = m_blockPrior.getBlockCount();
    auto flattenedEdgeMatrix = sampleRandomWeakComposition(
            m_edgeCountPrior.getState(),
            blockCount*(blockCount+1)/2
            );

    m_state = EdgeMatrix(blockCount, std::vector<size_t>(blockCount, 0));
    std::pair<BlockIndex, BlockIndex> rs;
    m_edgeCountsInBlocks = std::vector<size_t>(blockCount, 0);
    size_t index(0), correctedEdgeCount;
    for (auto edgeCountBetweenBlocks: flattenedEdgeMatrix) {
        rs = getUndirectedPairFromIndex(index, blockCount);
        m_edgeCountsInBlocks[rs.first] += edgeCountBetweenBlocks;
        m_edgeCountsInBlocks[rs.second] += edgeCountBetweenBlocks;
        correctedEdgeCount = rs.first!=rs.second ? edgeCountBetweenBlocks : 2*edgeCountBetweenBlocks;
        m_state[rs.first][rs.second] = correctedEdgeCount;
        m_state[rs.second][rs.first] = correctedEdgeCount;
        index++;
    }
}

double EdgeMatrixUniformPrior::getLogLikelihoodRatio(const GraphMove& move) const {
    double currentLogLikelihood =  getLogLikelihood(m_blockPrior.getBlockCount(), m_edgeCountPrior.getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPrior.getBlockCount(), m_edgeCountPrior.getState() + move.addedEdges.size() - move.removedEdges.size());
    return newLogLikelihood - currentLogLikelihood;
}

double EdgeMatrixUniformPrior::getLogLikelihoodRatio(const BlockMove& move) const {
    auto vertexCountsInBlocks = m_blockPrior.getVertexCountsInBlock();

    bool creatingBlock = move.nextBlockIdx == m_blockPrior.getBlockCount();
    bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx &&
                        vertexCountsInBlocks[move.prevBlockIdx] == 1;
    double currentLogLikelihood =  getLogLikelihood(m_blockPrior.getBlockCount(), m_edgeCountPrior.getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPrior.getBlockCount() + creatingBlock - destroyingBlock, m_edgeCountPrior.getState());
    return newLogLikelihood - currentLogLikelihood;
}

} // namespace FastMIDyNet
