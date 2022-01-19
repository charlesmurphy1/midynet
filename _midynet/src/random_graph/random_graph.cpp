#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/random_graph.h"

using namespace std;
using namespace BaseGraph;

namespace FastMIDyNet {

void RandomGraph::applyGraphMove(const GraphMove& move){
    for (auto edge: move.addedEdges){
        auto v = edge.first, u = edge.second;
        m_graph.addEdgeIdx(v, u);
    }
    for (auto edge: move.removedEdges){
        auto v = edge.first, u = edge.second;
        if ( m_graph.isEdgeIdx(u, v) )
            m_graph.removeEdgeIdx(v, u);
        else
            throw std::logic_error("Cannot remove non-existing edge (" + to_string(u) + ", " + to_string(v) + ").");
    }
}

const size_t RandomGraph::computeBlockCount() const {
    auto blocks = getBlocks();
    return *max_element(blocks.begin(), blocks.end()) + 1;
}

const std::vector<size_t> RandomGraph::computeVertexCountsInBlocks() const {
    auto blocks = getBlocks();
    std::vector<size_t> vertexCounts(getBlockCount(), 0);
    for (auto idx : m_graph){
        ++vertexCounts[blocks[idx]];
    }
    return vertexCounts;
}

const Matrix<size_t> RandomGraph::computeEdgeMatrix() const {
    auto blocks = getBlocks();
    auto blockCount = getBlockCount();
    Matrix<size_t> edgeMatrix(blockCount, {blockCount, 0});
    for (auto idx: m_graph){
        for(auto neighbor : m_graph.getNeighboursOfIdx(idx)){
            size_t edgeMult = neighbor.label;
            if (idx == neighbor.vertexIndex)
                edgeMult *= 2;
            edgeMatrix[blocks[idx]][blocks[neighbor.vertexIndex]] += neighbor.label;
        }
    }
    return edgeMatrix;
}

const std::vector<size_t> RandomGraph::computeEdgeCountsInBlocks() const {
    auto blockCount = getBlockCount();
    auto edgeMatrix = getEdgeMatrix();
    std::vector<size_t> edgeCounts(blockCount, 0);

    for (size_t blockIdx = 0; blockIdx < blockCount; ++blockIdx){
        for (auto ers : edgeMatrix[blockIdx]){
            edgeCounts[blockIdx] += ers;
        }
    }
    return edgeCounts;
}

const std::vector<CounterMap<size_t>> RandomGraph::computeDegreeCountsInBlocks() const {
    auto blockCount = getBlockCount();
    auto blocks = getBlocks();
    auto edgeMatrix = getEdgeMatrix();
    std::vector<CounterMap<size_t>> degreeCounts(blockCount);

    for(size_t idx: m_graph){
        size_t degree = m_graph.getDegreeOfIdx(idx);
        BlockIndex block = blocks[idx];
        degreeCounts[block].increment(degree);
    }

    return degreeCounts;

}

}
