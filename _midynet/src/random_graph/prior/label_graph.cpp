#include <string>
#include <vector>


#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"


namespace FastMIDyNet {

/* DEFINITION OF EDGE MATRIX PRIOR BASE CLASS */
void LabelGraphPrior::recomputeConsistentState() {
    m_edgeCounts = computeEdgeCountsFromState(m_state);
    m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
}
void LabelGraphPrior::recomputeStateFromGraph() {
    MultiGraph state(m_blockPriorPtr->getMaxBlockCount());
    for (const auto& vertex: *m_graphPtr){
        for (const auto& neighbor: m_graphPtr->getNeighboursOfIdx(vertex)){
            if (vertex > neighbor.vertexIndex)
                continue;

            state.addMultiedgeIdx(
                m_blockPriorPtr->getBlockOfIdx(vertex), m_blockPriorPtr->getBlockOfIdx(neighbor.vertexIndex), neighbor.label
            );
        }
    }
    setState(state);
}

void LabelGraphPrior::setGraph(const MultiGraph& graph) {
    m_graphPtr = &graph;
    recomputeStateFromGraph();
}

void LabelGraphPrior::setPartition(const std::vector<BlockIndex>& labels) {
    m_blockPriorPtr->setState(labels);
    recomputeStateFromGraph();
}


void LabelGraphPrior::setState(const MultiGraph& labelGraph) {
    m_state = labelGraph;
    recomputeConsistentState();
}

void LabelGraphPrior::applyLabelMoveToState(const BlockMove& move) {
    if (move.prevLabel == move.nextLabel)
        return;

    if (m_state.getSize() <= move.nextLabel)
        m_state.resize(move.nextLabel + 1);
    const auto& blockSeq = m_blockPriorPtr->getState();
    const auto& degree = m_graphPtr->getDegreeOfIdx(move.vertexIndex);

    m_edgeCounts.decrement(move.prevLabel, degree);
    m_edgeCounts.increment(move.nextLabel, degree);
    for (auto neighbor: m_graphPtr->getNeighboursOfIdx(move.vertexIndex)) {
        auto neighborBlock = blockSeq[neighbor.vertexIndex];

        if (move.vertexIndex == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.prevLabel;
        m_state.removeMultiedgeIdx(move.prevLabel, neighborBlock, neighbor.label) ;

        if (move.vertexIndex == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.nextLabel;
        m_state.addMultiedgeIdx(move.nextLabel, neighborBlock, neighbor.label) ;
    }

}

void LabelGraphPrior::applyGraphMoveToState(const GraphMove& move){
    const auto& blockSeq = m_blockPriorPtr->getState();

    for (auto removedEdge: move.removedEdges) {
        const BlockIndex& r(blockSeq[removedEdge.first]), s(blockSeq[removedEdge.second]);
        m_state.removeEdgeIdx(r, s);
        m_edgeCounts.decrement(r);
        m_edgeCounts.decrement(s);
    }
    for (auto addedEdge: move.addedEdges) {
        const BlockIndex& r(blockSeq[addedEdge.first]), s(blockSeq[addedEdge.second]);
        m_state.addEdgeIdx(r, s);
        m_edgeCounts.increment(r);
        m_edgeCounts.increment(s);
    }
}

void LabelGraphPrior::checkSelfConsistency() const {
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeCountPriorPtr->checkSelfConsistency();

    size_t sumEdges = 0;
    for (BlockIndex r : m_state) {
        size_t actualEdgeCounts = m_state.getDegreeOfIdx(r);
        if (actualEdgeCounts != m_edgeCounts[r])
            throw ConsistencyError("LabelGraphPrior: Edge matrix row ("
            + std::to_string(actualEdgeCounts) + ") doesn't sum to edgeCounts["
            + std::to_string(r) + "] (" + std::to_string(m_edgeCounts[r]) + ").");
        sumEdges += actualEdgeCounts;
    }
    if (sumEdges != 2*m_edgeCountPriorPtr->getState())
    throw ConsistencyError("LabelGraphPrior: Sum of edge matrix (" + std::to_string(sumEdges)
        + ") isn't equal to the number of edges (" + std::to_string(2*m_edgeCountPriorPtr->getState()) + ").");
}


const double LabelGraphDeltaPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
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

const double LabelGraphDeltaPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (move.prevLabel != move.nextLabel)
        return -INFINITY;
    return 0.;
}



/* DEFINITION OF EDGE MATRIX UNIFORM PRIOR */

void LabelGraphErdosRenyiPrior::sampleState() {
    LabelGraph labelGraph(m_blockPriorPtr->getBlockCount());

    std::vector<std::pair<BlockIndex, BlockIndex>> allEdges;

    for(auto nr : m_blockPriorPtr->getVertexCounts()){
        if (nr.second == 0)
            continue;
        for(auto ns : m_blockPriorPtr->getVertexCounts()){
            if (ns.second == 0 or nr.first > ns.first)
                continue;
            allEdges.push_back({nr.first, ns.first});
        }
    }

    auto edgeMultiplicities = sampleRandomWeakComposition(getEdgeCount(), allEdges.size());

    size_t counter = 0;
    for (auto m: edgeMultiplicities){
        if (m != 0)
            labelGraph.addMultiedgeIdx(allEdges[counter].first, allEdges[counter].second, m);
        ++counter;
    }

    setState(labelGraph);
}


const double LabelGraphErdosRenyiPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getEffectiveBlockCount(), m_edgeCountPriorPtr->getState());
    double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getEffectiveBlockCount(), m_edgeCountPriorPtr->getState() + move.addedEdges.size() - move.removedEdges.size());
    return newLogLikelihood - currentLogLikelihood;
}

const double LabelGraphErdosRenyiPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    double currentLogLikelihood =  getLogLikelihood(
        m_blockPriorPtr->getEffectiveBlockCount(), m_edgeCountPriorPtr->getState()
    );
    double newLogLikelihood =  getLogLikelihood(
        m_blockPriorPtr->getEffectiveBlockCount() + m_blockPriorPtr->getAddedBlocks(move), m_edgeCountPriorPtr->getState()
    );
    return newLogLikelihood - currentLogLikelihood;
}

// /* DEFINITION OF EDGE MATRIX EXPONENTIAL PRIOR */
//
// void LabelGraphExponentialPrior::sampleState() {
//     auto blockCount = m_blockPriorPtr->getBlockCount();
//     auto flattenedLabelGraph = sampleRandomWeakComposition(
//             m_edgeCountPriorPtr->getState(),
//             blockCount*(blockCount+1)/2
//             );
//
//     m_state = LabelGraph(blockCount, std::vector<size_t>(blockCount, 0));
//     std::pair<BlockIndex, BlockIndex> rs;
//     m_edgeCounts = std::vector<size_t>(blockCount, 0);
//     size_t index = 0, correctedEdgeCount;
//     for (auto edgeCountBetweenBlocks: flattenedLabelGraph) {
//         rs = getUndirectedPairFromIndex(index, blockCount);
//         m_edgeCounts[rs.first] += edgeCountBetweenBlocks;
//         m_edgeCounts[rs.second] += edgeCountBetweenBlocks;
//         m_state[rs.first][rs.second] += edgeCountBetweenBlocks;
//         m_state[rs.second][rs.first] += edgeCountBetweenBlocks;
//         index++;
//     }
// }
//
// double LabelGraphExponentialPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
//     double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
//     double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState() + move.addedEdges.size() - move.removedEdges.size());
//     return newLogLikelihood - currentLogLikelihood;
// }
//
// double LabelGraphExponentialPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
//     auto vertexCounts = m_blockPriorPtr->getVertexCounts();
//
//     bool creatingBlock = move.nextLabel == m_blockPriorPtr->getBlockCount();
//     bool destroyingBlock = move.nextLabel != move.prevLabel &&
//                         vertexCounts[move.prevLabel] == 1;
//     double currentLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
//     double newLogLikelihood =  getLogLikelihood(m_blockPriorPtr->getBlockCount() + creatingBlock - destroyingBlock, m_edgeCountPriorPtr->getState());
//     return newLogLikelihood - currentLogLikelihood;
// }



} // namespace FastMIDyNet
