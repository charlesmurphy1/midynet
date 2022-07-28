#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"

namespace FastMIDyNet{

void NestedLabelGraphPrior::applyGraphMoveToState(const GraphMove& move) {

    BlockIndex r, s;
    for (auto removedEdge: move.removedEdges) {
        r = (BlockIndex) removedEdge.first;
        s = (BlockIndex) removedEdge.second;
        for (Level l=0; l<getDepth(); ++l){
            r = m_nestedBlockPriorPtr->getNestedState(l)[r];
            s = m_nestedBlockPriorPtr->getNestedState(l)[s];
            m_nestedState[l].removeEdgeIdx(r, s);
            m_nestedEdgeCounts[l].decrement(r);
            m_nestedEdgeCounts[l].decrement(s);
        }
    }
    for (auto addedEdge: move.addedEdges) {
        r = (BlockIndex) addedEdge.first;
        s = (BlockIndex) addedEdge.second;
        for (Level l=0; l<getDepth(); ++l){
            r = m_nestedBlockPriorPtr->getNestedState(l)[r];
            s = m_nestedBlockPriorPtr->getNestedState(l)[s];
            m_nestedState[l].addEdgeIdx(r, s);
            m_nestedEdgeCounts[l].increment(r);
            m_nestedEdgeCounts[l].increment(s);
        }
    }
}

void NestedLabelGraphPrior::applyLabelMoveToState(const BlockMove& move) {
    // identity move
    if (move.prevLabel == move.nextLabel)
        return;

    // move creating new layer
    if (move.addedLabels == 1 and move.level == m_nestedState.size() - 1){
        m_nestedState.push_back({});
        m_nestedState[move.level + 1].resize(1);
        m_nestedState[move.level + 1].addMultiedgeIdx(0, 0, getEdgeCount());

        m_nestedEdgeCounts.push_back({});
        m_nestedEdgeCounts[move.level + 1].increment(0, 2 * getEdgeCount());
    }


    // move creating new label
    if (m_nestedState[move.level].getSize() <= move.nextLabel)
        m_nestedState[move.level].resize(move.nextLabel + 1);



    BlockIndex vertexIndex = getBlockOfIdx(move.vertexIndex, move.level-1);
    const LabelGraph& graph = getNestedState(move.level-1);
    const BlockSequence& blocks = getNestedBlocks(move.level);
    const auto& degree = graph.getDegreeOfIdx(vertexIndex);

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
            m_nestedEdgeCounts[l].set(r, m_nestedState[l].getDegreeOfIdx(r));
    m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
}

void NestedLabelGraphPrior::recomputeStateFromGraph() {
    BlockIndex r, s;
    for (Level l=0; l<getDepth(); ++l){
        m_nestedState[l] = MultiGraph(m_nestedBlockPriorPtr->getNestedMaxBlockCount(l));
        const LabelGraph& graph = getNestedState(l - 1);
        for (const auto& vertex: graph){
            for (const auto& neighbor: graph.getNeighboursOfIdx(vertex)){
                if (vertex > neighbor.vertexIndex)
                    continue;
                r = m_nestedBlockPriorPtr->getNestedState(l)[vertex];
                s = m_nestedBlockPriorPtr->getNestedState(l)[neighbor.vertexIndex];
                m_nestedState[l].addMultiedgeIdx(r, s, neighbor.label);
            }
        }
    }

    setNestedState(m_nestedState);
}

void NestedLabelGraphPrior::updateNestedEdgeDiffFromEdge(
    const BaseGraph::Edge& edge, std::vector<IntMap<BaseGraph::Edge>>& nestedEdgeDiff, int counter
) const{
    size_t r = edge.first, s = edge.second;
    for (Level l=0; l<getDepth(); ++l){
        r = getNestedBlockOfIdx(r, l);
        s = getNestedBlockOfIdx(s, l);
        nestedEdgeDiff[l].increment({r, s}, counter);
    }
}

void NestedLabelGraphPrior::sampleState() {
    m_nestedState = std::vector<LabelGraph>(getDepth());
    for (Level l=getDepth()-1; l>=0; --l){
        m_nestedState[l] = sampleState(l);
    }
    m_nestedEdgeCounts = computeNestedEdgeCountsFromNestedState(m_nestedState);
    m_state = m_nestedState[0];
    m_edgeCounts = m_nestedEdgeCounts[0];
}

const double NestedLabelGraphPrior::getLogLikelihood() const {
    double logLikelihood = 0;
    for (Level l=0; l<getDepth(); ++l)
        logLikelihood += getLogLikelihoodAtLevel(l);
    return logLikelihood;
}

void NestedLabelGraphPrior::checkSelfConsistencyBetweenLevels() const{
    LabelGraph graph, actualLabelGraph;
    BlockIndex r, s;
    std::string prefix;


    for (Level l=1; l<getDepth(); ++l){
        graph = getNestedState(l - 1);
        actualLabelGraph = LabelGraph(m_nestedBlockPriorPtr->getNestedMaxBlockCount(l));
        for (const auto& vertex: graph){
            for (const auto& neighbor: graph.getNeighboursOfIdx(vertex)){
                if (vertex > neighbor.vertexIndex)
                    continue;
                r = m_nestedBlockPriorPtr->getNestedState(l)[vertex];
                s = m_nestedBlockPriorPtr->getNestedState(l)[neighbor.vertexIndex];
                actualLabelGraph.addMultiedgeIdx(r, s, neighbor.label);
            }
        }
        prefix = "NestedLabelGraphPrior (level=" + std::to_string(l) + ")";
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

            if (actualLabelGraph.getDegreeOfIdx(vertex) != m_nestedState[l].getDegreeOfIdx(vertex))
                throw ConsistencyError(
                    prefix + ": graph is inconsistent with edge counts at (r="
                    + std::to_string(vertex)
                    + "): expected=" + std::to_string(m_nestedState[l].getDegreeOfIdx(vertex)) + ", actual="
                    + std::to_string(actualLabelGraph.getDegreeOfIdx(vertex))
                    + "."
                );
        }

    }

}

const LabelGraph NestedStochasticBlockLabelGraphPrior::sampleState(Level level) const {
    BlockSequence blocks;
    if (level == getDepth() - 1)
        blocks.push_back(0);
    else
        blocks = getNestedBlocks(level + 1);

    std::map<std::pair<BlockIndex, BlockIndex>, std::vector<BaseGraph::Edge>> allLabeledEdges;
    for (auto nr : m_nestedBlockPriorPtr->getNestedAbsVertexCounts(level)){
        if (nr.second == 0)
            continue;
        for (auto ns : m_nestedBlockPriorPtr->getNestedAbsVertexCounts(level)){
            if (ns.second == 0 or nr.first > ns.first)
                continue;
            auto rs = getOrderedPair<BlockIndex>({blocks[nr.first], blocks[ns.first]});
            allLabeledEdges[rs].push_back({nr.first, ns.first});
        }
    }

    LabelGraph graph(m_nestedBlockPriorPtr->getNestedBlockCount(level));
    for(const auto& labeledEdges : allLabeledEdges){
        BlockIndex r = labeledEdges.first.first, s = labeledEdges.first.second;
        size_t ers = (level == getDepth() - 1) ? getEdgeCount() : getNestedState(level + 1).getEdgeMultiplicityIdx(r, s);
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
const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodAtLevel(Level level) const {
    double logLikelihood = 0;
    size_t nr, ns, label;
    for (const auto& r : getNestedState(level)){
        nr = getNestedVertexCounts(level)[r];
        label = getNestedState(level).getEdgeMultiplicityIdx(r, r);
        logLikelihood -= logMultisetCoefficient(nr * (nr + 1) / 2, 2 * label);
        for (const auto& s : getNestedState(level)){
            if (r >= s)
                continue;
            ns = getNestedVertexCounts(level)[s];
            label = getNestedState(level).getEdgeMultiplicityIdx(r, s);
            logLikelihood -= logMultisetCoefficient(nr * ns, label);
        }
    }
    return logLikelihood;
}

const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
    std::vector<IntMap<BaseGraph::Edge>> nestedEdgeDiff(getDepth());
    for (auto edge : move.addedEdges)
        updateNestedEdgeDiffFromEdge(edge, nestedEdgeDiff, 1);
    for (auto edge : move.removedEdges)
        updateNestedEdgeDiffFromEdge(edge, nestedEdgeDiff, -1);

    double logLikelihoodRatio = 0;
    for (Level l=0; l<getDepth(); ++l){
        size_t vTerm, eTermBefore, eTermAfter, nr, ns;
        for (auto diff: nestedEdgeDiff[l]){
            BlockIndex r = diff.first.first, s = diff.first.second;
            nr = getNestedVertexCounts(l)[r], ns = getNestedVertexCounts(l)[s];
            vTerm = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
            eTermBefore = ((r == s) ? 2 : 1) * getNestedState(l).getEdgeMultiplicityIdx(r, s);
            eTermAfter = ((r == s) ? 2 : 1) * (getNestedState(l).getEdgeMultiplicityIdx(r, s) + diff.second);

            logLikelihoodRatio -= logMultisetCoefficient(vTerm, eTermAfter) - logMultisetCoefficient(vTerm, eTermBefore);
        }
    }
    return logLikelihoodRatio;
}

const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    BlockIndex nestedIndex = m_nestedBlockPriorPtr->getBlockOfIdx(move.vertexIndex, move.level-1);
    IntMap<BaseGraph::Edge> edgeDiff;
    IntMap<BlockIndex> vertexDiff;
    vertexDiff.decrement(move.prevLabel);
    vertexDiff.increment(move.nextLabel);

    for (auto neighbor : getNestedState(move.level-1).getNeighboursOfIdx(nestedIndex)){
        BlockIndex s = m_nestedBlockPriorPtr->getNestedBlockOfIdx(neighbor.vertexIndex, move.level);
        if (neighbor.vertexIndex == nestedIndex)
            s = move.prevLabel;
        edgeDiff.decrement(getOrderedEdge({move.prevLabel, s}), neighbor.label);
        if (neighbor.vertexIndex == nestedIndex)
            s = move.nextLabel;
        edgeDiff.increment(getOrderedEdge({move.nextLabel, s}), neighbor.label);
    }
    double logLikelihoodRatio = 0;
    size_t vTermBefore, vTermAfter, eTermBefore, eTermAfter, nr, ns, edgeMult;
    BlockIndex r, s;

    // contributions that changed the edge counts
    for (auto diff: edgeDiff){
        r = diff.first.first, s = diff.first.second;
        nr = getNestedVertexCounts(move.level)[r], ns = getNestedVertexCounts(move.level)[s];
        vTermBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
        vTermAfter = (r == s) ? (nr + vertexDiff.get(r)) * (nr + vertexDiff.get(r) + 1) / 2 : (nr + vertexDiff.get(r)) * (ns + vertexDiff.get(s));
        edgeMult = (move.addedLabels == 1 and (r == move.nextLabel or s == move.nextLabel)) ? 0 : getNestedState(move.level).getEdgeMultiplicityIdx(r, s);
        eTermBefore = ((r == s) ? 2 : 1) * edgeMult;
        eTermAfter = ((r == s) ? 2 : 1) * (edgeMult + diff.second);

        logLikelihoodRatio -= logMultisetCoefficient(vTermAfter, eTermAfter) - logMultisetCoefficient(vTermBefore, eTermBefore);
    }

    // remaining contributions that did not change the edge counts
    std::set<BaseGraph::Edge> visited;
    for (const auto& diff : vertexDiff){
        if (move.addedLabels == 1 and diff.first == move.nextLabel)
            continue;
        for (const auto& neighbor : getNestedState(move.level).getNeighboursOfIdx(diff.first)){
            r = diff.first, s = neighbor.vertexIndex;
            // if not empty, the term has been processed in the edgeDiff loop
            auto rs = getOrderedEdge({r, s});
            if (visited.count(rs) > 0 or not edgeDiff.isEmpty(rs))
                continue;
            visited.insert(rs);
            nr = getNestedVertexCounts(move.level)[r], ns = getNestedVertexCounts(move.level)[s];
            vTermBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
            vTermAfter = (r == s) ? (nr + vertexDiff.get(r)) * (nr + vertexDiff.get(r) + 1) / 2 : (nr + vertexDiff.get(r)) * (ns + vertexDiff.get(s));
            edgeMult = ((r == s) ? 2 : 1) * neighbor.label;
            logLikelihoodRatio -= logMultisetCoefficient(vTermAfter, edgeMult) - logMultisetCoefficient(vTermBefore, edgeMult);
        }
    }

    if (move.addedLabels == 1){
        // if adding new label not in last layer
        if (move.level < getDepth() - 1){
            r = getNestedBlockOfIdx(move.prevLabel, move.level + 1);
            nr = getNestedVertexCounts(move.level + 1)[r];
            for (const auto& neighbor : getNestedState(move.level + 1).getNeighboursOfIdx(r)){
                s = neighbor.vertexIndex;
                ns = getNestedVertexCounts(move.level + 1)[s];
                vTermBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
                vTermAfter = (r == s) ? (nr + 1) * (nr + 2) / 2 : (nr + 1) * ns;
                eTermBefore = ((r == s) ? 2 : 1) * neighbor.label;
                logLikelihoodRatio -= logMultisetCoefficient(vTermAfter, eTermBefore) - logMultisetCoefficient(vTermBefore, eTermBefore);
            }
        // if adding new label in last layer
        } else {
            nr = getNestedVertexCounts(move.level).size() + 1;
            logLikelihoodRatio -= logMultisetCoefficient(nr * (nr + 1) / 2, 2 * getEdgeCount());
        }
    }
    return logLikelihoodRatio;
}




}
