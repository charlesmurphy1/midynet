#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"

namespace FastMIDyNet{

void NestedLabelGraphPrior::applyGraphMoveToState(const GraphMove& move) {

    BlockIndex r, s;
    for (auto removedEdge: move.removedEdges) {
        r = (BlockIndex) removedEdge.first;
        s = (BlockIndex) removedEdge.second;
        for (Level l=0; l<m_nestedBlockPriorPtr->getDepth(); ++l){
            r = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[r];
            s = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[s];
            m_nestedState[l].removeEdgeIdx(r, s);
            m_nestedEdgeCounts[l].decrement(r);
            m_nestedEdgeCounts[l].decrement(s);
        }
    }
    for (auto addedEdge: move.addedEdges) {
        r = (BlockIndex) addedEdge.first;
        s = (BlockIndex) addedEdge.second;
        for (Level l=0; l<m_nestedBlockPriorPtr->getDepth(); ++l){
            r = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[r];
            s = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[s];
            m_nestedState[l].addEdgeIdx(r, s);
            m_nestedEdgeCounts[l].increment(r);
            m_nestedEdgeCounts[l].increment(s);
        }
    }
}

void NestedLabelGraphPrior::applyLabelMoveToState(const BlockMove& move) {
    if (move.prevLabel == move.nextLabel)
        return;
    BlockIndex vertexIndex = m_nestedBlockPriorPtr->getBlockOfIdx(move.vertexIndex, move.level-1);
    const MultiGraph& graph = getNestedStateAtLevel(move.level-1);
    const BlockSequence& blocks = m_nestedBlockPriorPtr->getNestedStateAtLevel(move.level);
    const auto& degree = graph.getDegreeOfIdx(vertexIndex);

    if (m_nestedBlockPriorPtr->creatingNewLevel(move)){
        m_nestedState.push_back({});
        m_nestedEdgeCounts.push_back({});
    }

    if (m_nestedState[move.level].getSize() <= move.nextLabel)
        m_nestedState[move.level].resize(move.nextLabel + 1);


    m_nestedEdgeCounts[move.level].decrement(move.prevLabel, degree);
    m_nestedEdgeCounts[move.level].increment(move.nextLabel, degree);
    for (auto neighbor: graph.getNeighboursOfIdx(vertexIndex)) {
        auto neighborBlock = blocks[neighbor.vertexIndex];

        if (vertexIndex == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.prevLabel;
        m_nestedState[move.level].removeMultiedgeIdx(move.prevLabel, neighborBlock, neighbor.label) ;

        if (vertexIndex == neighbor.vertexIndex) // for self-loops
            neighborBlock = move.nextLabel;
        m_nestedState[move.level].addMultiedgeIdx(move.nextLabel, neighborBlock, neighbor.label) ;
    }
}


void NestedLabelGraphPrior::recomputeConsistentState() {
    m_nestedEdgeCounts.clear();
    m_nestedEdgeCounts.resize(m_nestedBlockPriorPtr->getDepth(), {});
    for (Level l=0; l<m_nestedBlockPriorPtr->getDepth(); ++l)
        for (auto r : m_nestedState[l])
            m_nestedEdgeCounts[l].set(r, m_state.getDegreeOfIdx(r));
    m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
}

void NestedLabelGraphPrior::recomputeStateFromGraph() {
    std::vector<MultiGraph> nestedState(m_nestedBlockPriorPtr->getDepth());
    BlockIndex r, s;
    for (Level l=0; l<m_nestedBlockPriorPtr->getDepth(); ++l){
        nestedState[l].resize(m_nestedBlockPriorPtr->getNestedMaxBlockCountAtLevel(l));
        const MultiGraph& graph = getNestedStateAtLevel(l);
        for (const auto& vertex: graph){
            for (const auto& neighbor: graph.getNeighboursOfIdx(vertex)){
                if (vertex > neighbor.vertexIndex)
                    continue;
                r = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[vertex];
                s = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[neighbor.vertexIndex];
                nestedState[l].addMultiedgeIdx(r, s, neighbor.label);
            }
        }
    }
    setNestedState(nestedState);
}

void NestedLabelGraphPrior::checkSelfConsistencyBetweenLevels() const{
    MultiGraph graph, actualLabelGraph;
    BlockIndex r, s;
    std::string prefix;

    for (Level l=0; l<m_nestedBlockPriorPtr->getDepth(); ++l){
        graph = getNestedStateAtLevel(l - 1);
        actualLabelGraph = MultiGraph(m_nestedBlockPriorPtr->getNestedMaxBlockCountAtLevel(l));
        prefix = "NestedLabelGraphPrior (level=" + std::to_string(l) + ")";

        for (const auto& vertex: graph){
            for (const auto& neighbor: graph.getNeighboursOfIdx(vertex)){
                if (vertex > neighbor.vertexIndex)
                    continue;
                r = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[vertex];
                s = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[neighbor.vertexIndex];
                actualLabelGraph.addMultiedgeIdx(r, s, neighbor.label);
            }
        }

        for (const auto& vertex: m_nestedState[l]){
            for (const auto& neighbor: m_nestedState[l].getNeighboursOfIdx(vertex)){
                if (vertex > neighbor.vertexIndex)
                    continue;
                if (actualLabelGraph.getEdgeMultiplicityIdx(vertex, neighbor.vertexIndex) != neighbor.label)
                    throw ConsistencyError(
                        prefix + ": graph is inconsistent with label graph at (r="
                        + std::to_string(vertex) + ", s=" + std::to_string(neighbor.vertexIndex)
                        + "): expected=" + std::to_string(neighbor.label) + ", actual="
                        + std::to_string(actualLabelGraph.getEdgeMultiplicityIdx(vertex, neighbor.vertexIndex))
                        + "."
                    );
            }

            if (actualLabelGraph.getDegreeOfIdx(vertex) != m_nestedEdgeCounts[l][vertex])
                throw ConsistencyError(
                    prefix + ": graph is inconsistent with edge counts at (r="
                    + std::to_string(vertex)
                    + "): expected=" + std::to_string(m_nestedEdgeCounts[l][vertex]) + ", actual="
                    + std::to_string(actualLabelGraph.getDegreeOfIdx(vertex))
                    + "."
                );
        }

    }

}


}
