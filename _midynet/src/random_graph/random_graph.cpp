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

void RandomGraph::_applyGraphMove(const GraphMove& move){
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

// const size_t RandomGraph::computeBlockCount() const {
//     auto blocks = getBlocks();
//     return *max_element(blocks.begin(), blocks.end()) + 1;
// }

// const CounterMap<size_t> RandomGraph::computeVertexCountsInBlocks() const {
//     auto blocks = getBlocks();
//     CounterMap<size_t> vertexCounts;
//     for (auto idx : blocks){
//         vertexCounts.increment(idx);
//     }
//     return vertexCounts;
// }
//
// const Matrix<size_t> RandomGraph::computeEdgeMatrix() const {
//     auto blocks = getBlocks();
//     auto blockCount = getBlockCount();
//     Matrix<size_t> edgeMatrix(blockCount, {blockCount, 0});
//     for (auto idx: m_graph){
//         for(auto neighbor : m_graph.getNeighboursOfIdx(idx)){
//             size_t edgeMult = neighbor.label;
//             if (idx == neighbor.vertexIndex)
//                 edgeMult *= 2;
//             edgeMatrix[blocks[idx]][blocks[neighbor.vertexIndex]] += neighbor.label;
//         }
//     }
//     return edgeMatrix;
// }

const DegreeCountsMap RandomGraph::computeDegreeCountsInBlocks() const {
    auto blockCount = getBlockCount();
    auto blocks = getBlocks();
    auto edgeMatrix = getEdgeMatrix();
    DegreeCountsMap degreeCounts;

    for(size_t idx: m_graph)
        degreeCounts.increment({blocks[idx], m_graph.getDegreeOfIdx(idx)});

    return degreeCounts;

}

}
