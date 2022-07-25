#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"

namespace FastMIDyNet{

// BlockSequence remapEmptyBlocks(const BlockSequence& blocks){
//     CounterMap<BlockIndex> counts;
//     for (auto b : blocks)
//         counts.increment(b);
//
//     std::map<BlockIndex, BlockIndex> remapping;
//     BlockIndex id=0;
//     for (auto nr : counts){
//         if (nr.second > 0){
//             remapping.insert({nr.first, id});
//             ++id;
//         }
//     }
//
//     BlockSequence remappedBlocks;
//      for (size_t i=0; i<blocks.size(); ++i)
//         remappedBlocks.push_back(remapping.at(blocks[i]));
//     return remappedBlocks;
// }

void NestedLabelGraphPrior::applyGraphMoveToState(const GraphMove& move) {

    BlockIndex r, s;
    for (auto removedEdge: move.removedEdges) {
        r = (BlockIndex) removedEdge.first;
        s = (BlockIndex) removedEdge.second;
        for (Level l=0; l<getDepth(); ++l){
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
        for (Level l=0; l<getDepth(); ++l){
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
    const LabelGraph& graph = getNestedStateAtLevel(move.level-1);
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
    m_nestedEdgeCounts.resize(getDepth(), {});
    for (Level l=0; l<getDepth(); ++l)
        for (auto r : m_nestedState[l])
            m_nestedEdgeCounts[l].set(r, m_state.getDegreeOfIdx(r));
    m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
}

void NestedLabelGraphPrior::recomputeStateFromGraph() {
    std::vector<LabelGraph> nestedState(getDepth());
    BlockIndex r, s;
    for (Level l=0; l<getDepth(); ++l){
        nestedState[l].resize(m_nestedBlockPriorPtr->getNestedMaxBlockCountAtLevel(l));
        const LabelGraph& graph = getNestedStateAtLevel(l);
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


void NestedLabelGraphPrior::sampleState() {
    m_nestedState = std::vector<LabelGraph>(getDepth());
    for (Level l=getDepth()-1; l>=0; --l){
        m_nestedState[l] = sampleStateAtLevel(l);
    }
    m_nestedEdgeCounts = computeNestedEdgeCountsFromNestedState(m_nestedState);
    m_state = m_nestedState[0];
    m_edgeCounts = m_nestedEdgeCounts[0];
}

void NestedLabelGraphPrior::checkSelfConsistencyBetweenLevels() const{
    LabelGraph graph, actualLabelGraph;
    BlockIndex r, s;
    std::string prefix;


    for (Level l=1; l<getDepth(); ++l){
        graph = getNestedStateAtLevel(l - 1);
        actualLabelGraph = LabelGraph(m_nestedBlockPriorPtr->getNestedMaxBlockCountAtLevel(l));
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

const LabelGraph NestedStochasticBlockLabelGraphPrior::sampleStateAtLevel(Level level) const {


    BlockSequence blocks;
    if (level == getDepth() - 1)
        blocks.push_back(0);
    else
        blocks = getNestedBlocksAtLevel(level + 1);

    std::map<std::pair<BlockIndex, BlockIndex>, std::vector<BaseGraph::Edge>> allLabeledEdges;
    for (auto nr : m_nestedBlockPriorPtr->getNestedVertexCountsAtLevel(level)){
        if (nr.second == 0)
            continue;
        for (auto ns : m_nestedBlockPriorPtr->getNestedVertexCountsAtLevel(level)){
            if (ns.second == 0 or nr.first > ns.first)
                continue;
            auto rs = getOrderedPair<BlockIndex>({blocks[nr.first], blocks[ns.first]});
            allLabeledEdges[rs].push_back({nr.first, ns.first});
        }
    }

    LabelGraph graph(m_nestedBlockPriorPtr->getNestedBlockCountAtLevel(level));
    for(const auto& labeledEdges : allLabeledEdges){
        BlockIndex r = labeledEdges.first.first, s = labeledEdges.first.second;
        size_t ers = (level == getDepth() - 1) ? getEdgeCount() : getNestedStateAtLevel(level + 1).getEdgeMultiplicityIdx(r, s);
        auto flatMultiplicity = sampleRandomWeakComposition(ers, labeledEdges.second.size());
        size_t counter = 0;
        for (const auto& m: flatMultiplicity){
            if (m != 0)
                graph.addMultiedgeIdx(labeledEdges.second[counter].first, labeledEdges.second[counter].second, m);
            ++counter;
        }
    }
    return graph;
}

double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioOfLevel(
        const CounterMap<BlockIndex>& vertexCounts,
        const LabelGraph& nextLabelGraph,
        const IntMap<BaseGraph::Edge>& edgeDiff,
        const IntMap<BlockIndex>& vertexDiff) const {
    double logLikelihoodRatio = 0;
    size_t vertexBefore, vertexAfter, edgeBefore, edgeAfter, nr, ns;
    BlockIndex r, s;

    for (auto diff: edgeDiff){
        r = diff.first.first, s = diff.first.second;
        nr = vertexCounts[r], ns = vertexCounts[s];
        vertexBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
        vertexAfter = (r == s) ? (nr + vertexDiff.get(r)) * (nr + vertexDiff.get(r) + 1) / 2 : (nr + vertexDiff.get(r)) * (ns + vertexDiff.get(s));
        edgeBefore = ((r == s) ? 2 : 1) * nextLabelGraph.getEdgeMultiplicityIdx(r, s);
        edgeAfter = ((r == s) ? 2 : 1) * (nextLabelGraph.getEdgeMultiplicityIdx(r, s) + diff.second);
        logLikelihoodRatio -= logMultisetCoefficient(vertexAfter, edgeAfter) - logMultisetCoefficient(vertexBefore, edgeBefore);
    }

    return logLikelihoodRatio;
}

double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodOfLevel( const CounterMap<BlockIndex>& vertexCounts, const LabelGraph& nextLabelGraph) const {
    double logLikelihood = 0;
    size_t nr, ns;
    for (const auto& r : nextLabelGraph){
        nr = vertexCounts[r];
        logLikelihood -= logMultisetCoefficient(nr * (nr + 1) / 2, 2 * nextLabelGraph.getEdgeMultiplicityIdx(r, r));
        for (const auto& s : nextLabelGraph.getNeighboursOfIdx(r)){
            if (r >= s.vertexIndex)
                continue;
            ns = vertexCounts[s.vertexIndex];
            logLikelihood -= logMultisetCoefficient(nr * ns, s.label);
        }
    }
    return logLikelihood;
}

const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    std::vector<IntMap<BaseGraph::Edge>> nestedEdgeDiff(getDepth());
    for (auto edge : move.addedEdges){
        updateNestedEdgeDiffFromEdge(edge, nestedEdgeDiff, 1);
    }
    for (auto edge : move.removedEdges){
        updateNestedEdgeDiffFromEdge(edge, nestedEdgeDiff, -1);
    }
    double logLikelihoodRatio = 0;
    for (Level l=0; l<getDepth(); ++l){
        logLikelihoodRatio += getLogLikelihoodRatioOfLevel(
                                m_nestedBlockPriorPtr->getNestedVertexCountsAtLevel(l),
                                getNestedStateAtLevel(l + 1),
                                nestedEdgeDiff[l], {}
                            );
    }
    return logLikelihoodRatio;

}

const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (m_nestedBlockPriorPtr->getNestedStateAtLevel(move.level + 1)[move.prevLabel] != m_nestedBlockPriorPtr->getNestedStateAtLevel(move.level + 1)[move.nextLabel])
        return -INFINITY;

    BlockIndex nestedIndex = m_nestedBlockPriorPtr->getBlockOfIdx(move.vertexIndex, move.level-1);
    IntMap<BaseGraph::Edge> edgeDiff;
    IntMap<BlockIndex> vertexDiff;
    vertexDiff.decrement(move.prevLabel);
    vertexDiff.increment(move.nextLabel);
    for (auto neighbor : getNestedStateAtLevel(move.level-1).getNeighboursOfIdx(nestedIndex)){
        BlockIndex s = m_nestedBlockPriorPtr->getNestedStateAtLevel(move.level)[neighbor.vertexIndex];
        edgeDiff.decrement(getOrderedEdge({move.prevLabel, s}));
        edgeDiff.increment(getOrderedEdge({move.nextLabel, s}));
    }
    return getLogLikelihoodRatioOfLevel(
                            m_nestedBlockPriorPtr->getNestedVertexCountsAtLevel(move.level),
                            getNestedStateAtLevel(move.level + 1),
                            edgeDiff, vertexDiff);
}


}
