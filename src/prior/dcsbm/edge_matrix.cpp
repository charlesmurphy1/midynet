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
    m_edgesInBlock = std::vector<size_t>(blockCount, 0);

    for (auto vertex: graph) {
        const BlockIndex& r(vertexBlocks[vertex]);
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex > neighbor.first)
                continue;

            const BlockIndex& s(vertexBlocks[neighbor.first]);
            m_edgesInBlock[r]+=neighbor.second;
            m_edgesInBlock[s]+=neighbor.second;
            m_state[r][s]+=neighbor.second;
            m_state[s][r]+=neighbor.second;
        }
    }
}

void EdgeMatrixPrior::createBlock() {
    const auto& blockCount = m_blockPrior.getBlockCount();
    for (auto& row: m_state)
        row.push_back(0);
    m_state.push_back(std::vector<size_t>(blockCount, 0));
}

void EdgeMatrixPrior::destroyBlock(const BlockIndex& block) {
    for (auto& row: m_state)
        row.erase(row.begin()+block);
    m_state.erase(m_state.begin()+block);
}

void EdgeMatrixPrior::moveEdgesInBlocks(const BlockMove& move) {
    const auto& vertexBlocks = m_blockPrior.getState();
    BlockIndex neighborBlock;

    for (auto neighbor: m_graph->getNeighboursOfIdx(move.vertexIdx)) {
        neighborBlock = vertexBlocks[neighbor.first];

        m_state[move.prevBlockIdx][neighborBlock] -= neighbor.second;
        m_state[neighborBlock][move.prevBlockIdx] -= neighbor.second;
        m_edgesInBlock[move.prevBlockIdx] -= neighbor.second;

        m_state[move.nextBlockIdx][neighborBlock] += neighbor.second;
        m_state[neighborBlock][move.nextBlockIdx] += neighbor.second;
        m_edgesInBlock[move.nextBlockIdx] += neighbor.second;
    }
}

template<typename T>
static void verifyVectorHasSize(const std::vector<T>& vec, size_t size,
        const std::string& vectorName, const std::string& sizeName) {
    if (vec.size() != size)
        throw ConsistencyError("EdgeMatrixPrior: "+vectorName+" has size "+
                std::to_string(vec.size())+" while there are "+
                std::to_string(size)+" "+sizeName+".");
}

void EdgeMatrixPrior::checkSelfConsistency() const {
    m_blockPrior.checkSelfConsistency();
    m_edgeCountPrior.checkSelfConsistency();

    const size_t& blockCount = m_blockPrior.getBlockCount();
    verifyVectorHasSize(m_edgesInBlock, blockCount, "edgesInBlock", "blocks");
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
        if (actualEdgesInBlock != m_edgesInBlock[i])
            throw ConsistencyError("EdgeMatrixPrior: Edge matrix row doesn't sum to edgesInBlock.");
        sumEdges += actualEdgesInBlock;
    }
    if (sumEdges != 2*m_edgeCountPrior.getState())
        throw ConsistencyError("EdgeMatrixPrior: Sum of edge matrix isn't equal to twice the number of edges.");
}



void EdgeMatrixUniformPrior::sampleState() {
    auto blockCount = m_blockPrior.getBlockCount();
    auto flattenedEdgeMatrix = sampleRandomWeakComposition(
            m_edgeCountPrior.getState(),
            blockCount*(blockCount+1)/2
            );

    m_state = Matrix<size_t>(blockCount, std::vector<size_t>(blockCount));
    std::pair<BlockIndex, BlockIndex> rs;
    BlockIndex& r(rs.first), s(rs.second);

    m_edgesInBlock = std::vector<size_t>(blockCount, 0);
    size_t index(0), correctedEdgeCount;
    for (auto edgeCountBetweenBlocks: flattenedEdgeMatrix) {
        rs = getUndirectedPairFromIndex(index, blockCount);
        m_edgesInBlock[r] += edgeCountBetweenBlocks;
        m_edgesInBlock[s] += edgeCountBetweenBlocks;

        correctedEdgeCount = r!=s ? edgeCountBetweenBlocks : 2*edgeCountBetweenBlocks;
        m_state[r][s] = correctedEdgeCount;
        m_state[s][r] = correctedEdgeCount;
        index++;
    }
}

double EdgeMatrixUniformPrior::getLogLikelihoodRatio(const GraphMove& move) const {
    return getLogLikelihood(m_blockPrior.getBlockCount(),
                        m_edgeCountPrior.getState()+move.addedEdges.size()-move.removedEdges.size());
}

double EdgeMatrixUniformPrior::getLogLikelihoodRatio(const BlockMove& move) const {
    auto vertexCountInBlocks = m_blockPrior.getVertexCountInBlock();

    bool creatingBlock = move.nextBlockIdx == m_blockPrior.getBlockCount();
    bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx &&
                        vertexCountInBlocks[move.prevBlockIdx] == 1;

    return getLogLikelihood(m_blockPrior.getBlockCount() + creatingBlock - destroyingBlock,
                        m_edgeCountPrior.getState());
}


void EdgeMatrixUniformPrior::applyMove(const GraphMove& move) {
    const auto& vertexBlocks = m_blockPrior.getState();

    for (auto addedEdge: move.addedEdges) {
        const BlockIndex& r(vertexBlocks[addedEdge.first]), s(vertexBlocks[addedEdge.second]);
        m_state[r][s]++;
        m_state[s][r]++;
        m_edgesInBlock[r]++;
        m_edgesInBlock[s]++;
    }
    for (auto removedEdge: move.removedEdges) {
        const BlockIndex& r(vertexBlocks[removedEdge.first]), s(vertexBlocks[removedEdge.second]);
        m_state[r][s]--;
        m_state[s][r]--;
        m_edgesInBlock[r]--;
        m_edgesInBlock[s]--;
    }
}

void EdgeMatrixUniformPrior::applyMove(const BlockMove& move) {
    auto vertexCountInBlocks = m_blockPrior.getVertexCountInBlock();

    bool creatingBlock = move.nextBlockIdx == m_blockPrior.getBlockCount();
    bool destroyingBlock = move.nextBlockIdx != move.prevBlockIdx &&
                        vertexCountInBlocks[move.prevBlockIdx] == 1;


    if (creatingBlock && destroyingBlock)
        return;
    if (creatingBlock)
        createBlock();

    moveEdgesInBlocks(move);

    if (destroyingBlock)
        destroyBlock(move.prevBlockIdx);
}

} // namespace FastMIDyNet
