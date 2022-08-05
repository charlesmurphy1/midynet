#ifndef FAST_MIDYNET_NESTED_MIXED_PROPOSER_H
#define FAST_MIDYNET_NESTED_MIXED_PROPOSER_H


#include "SamplableSet.hpp"

#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/nested_label/base.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"

namespace FastMIDyNet {

template<typename Label>
class MixedNestedSampler{
protected:
    double m_shift;
    const NestedVertexLabeledRandomGraph<Label>** m_nestedGraphPriorPtrPtr = nullptr;
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);
//
    const Label sampleNeighborLabelAtLevel(BaseGraph::VertexIndex vertex, Level level) const ;

    virtual const Label sampleLabelUniformlyAtLevel(Level level) const = 0;

    const Label sampleLabelPreferentiallyAtLevel(const Label neighborLabel, Level level) const ;
    const double _getLogProposalProbForMove(const LabelMove<Label>& move) const ;
    const double _getLogProposalProbForReverseMove(const LabelMove<Label>& move) const ;
    const LabelMove<Label> _proposeLabelMoveAtLevel(const BaseGraph::VertexIndex&, Level) const ;
    IntMap<std::pair<Label, Label>> getEdgeMatrixDiff(const LabelMove<Label>& move) const ;
    IntMap<Label> getEdgeCountsDiff(const LabelMove<Label>& move) const ;
    virtual const size_t getAvailableLabelCountAtLevel(Level) const = 0;

    // bool creatingNewLabel(const LabelMove<Label>& move) const {
    //     return (*m_nestedGraphPriorPtrPtr)->getNestedLabelCounts(move.level).get(move.nextLabel) == 0;
    // };
    // bool destroyingLabel(const LabelMove<Label>& move) const {
    //     return move.prevLabel != move.nextLabel and (*m_nestedGraphPriorPtrPtr)->getLabelCounts(move.level).get(move.prevLabel) == 1 ;
    // }
 public:
     MixedNestedSampler(double shift=1): m_shift(shift) {}

    const double getShift() const { return m_shift; }
};

template<typename Label>
const Label MixedNestedSampler<Label>::sampleNeighborLabelAtLevel(BaseGraph::VertexIndex vertex, Level level) const {
    Label index = (*m_nestedGraphPriorPtrPtr)->getLabelOfIdx(vertex, level - 1);
    const LabelGraph& labelGraph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(level-1);
    size_t degree = labelGraph.getDegreeOfIdx(index) - 2 * labelGraph.getEdgeMultiplicityIdx(index, index);
    size_t counter = std::uniform_int_distribution<size_t>(0, degree-1)(rng);
    for (const auto& neighbor : labelGraph.getNeighboursOfIdx(index)){
        if (neighbor.vertexIndex == index)
            continue;
        counter -= neighbor.label;
        if (counter < 0)
            return (*m_nestedGraphPriorPtrPtr)->getNestedLabelOfIdx(neighbor.vertexIndex, level);
    }
    return  (*m_nestedGraphPriorPtrPtr)->getNestedLabelOfIdx(index, level);
}

template<typename Label>
const Label MixedNestedSampler<Label>::sampleLabelPreferentiallyAtLevel(
    const Label neighborLabel, Level level
) const {
    std::uniform_int_distribution<int> dist(
        0, (*m_nestedGraphPriorPtrPtr)->getNestedEdgeLabelCounts(level).get(neighborLabel) - 1
    );
    const LabelGraph& labelGraph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(level);
    int counter = dist(rng);
    for (auto s : labelGraph.getNeighboursOfIdx(neighborLabel)){
        counter -= ((neighborLabel == s.vertexIndex) ? 2 : 1) * s.label;
        if (counter < 0)
            return s.vertexIndex;
    }
    return sampleLabelUniformlyAtLevel(level);
}
//
template<typename Label>
const LabelMove<Label> MixedNestedSampler<Label>::_proposeLabelMoveAtLevel(const BaseGraph::VertexIndex&vertex, Level level) const {
    const auto& edgeCounts = (*m_nestedGraphPriorPtrPtr)->getNestedEdgeLabelCounts(level);
    const auto& B = getAvailableLabelCountAtLevel(level);
    Label neighborLabel = MixedNestedSampler<Label>::sampleNeighborLabelAtLevel(vertex, level);
    double probUniformSampling = m_shift * B / (edgeCounts.get(neighborLabel) + m_shift * B);
    Label nextLabel = (m_uniform01(rng) < probUniformSampling) ? sampleLabelUniformlyAtLevel(level) : sampleLabelPreferentiallyAtLevel(neighborLabel, level);
    return {vertex, (*m_nestedGraphPriorPtrPtr)->getLabelOfIdx(vertex, level), nextLabel, 0, level};
}

template<typename Label>
const double MixedNestedSampler<Label>::_getLogProposalProbForMove(const LabelMove<Label>& move) const {
    const auto & labels = (*m_nestedGraphPriorPtrPtr)->getNestedLabels(move.level);
    const auto & edgeCounts = (*m_nestedGraphPriorPtrPtr)->getNestedEdgeLabelCounts(move.level);
    const auto & graph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(move.level-1);
    const auto &labelGraph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(move.level);
    BlockIndex index = (*m_nestedGraphPriorPtrPtr)->getLabelOfIdx(move.vertexIndex, move.level-1);

    double weight = 0, degree = 0;
    for (auto neighbor : graph.getNeighboursOfIdx(index)){
        if (index == neighbor.vertexIndex)
            continue;
        auto t = labels[ neighbor.vertexIndex ];
        size_t m = (move.nextLabel >= labelGraph.getSize()) ? 0 : labelGraph.getEdgeMultiplicityIdx(t, move.nextLabel);
        size_t Est = ((t == move.nextLabel) ? 2 : 1) * m;
        size_t Et = edgeCounts.get(t);

        degree += neighbor.label;
        weight += neighbor.label * ( Est + m_shift ) / (Et + m_shift * getAvailableLabelCountAtLevel(move.level)) ;
    }

    if (degree == 0)
       return -log(getAvailableLabelCountAtLevel(move.level));
    double logProposal = log(weight) - log(degree);
    return logProposal;
}

template<typename Label>
const double MixedNestedSampler<Label>::_getLogProposalProbForReverseMove(const LabelMove<Label>& move) const {
    const auto & labels = (*m_nestedGraphPriorPtrPtr)->getNestedLabels(move.level);
    const auto & edgeCounts = (*m_nestedGraphPriorPtrPtr)->getNestedEdgeLabelCounts(move.level);
    const auto & graph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(move.level-1);
    const auto &labelGraph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(move.level);
    BlockIndex index = (*m_nestedGraphPriorPtrPtr)->getLabelOfIdx(move.vertexIndex, move.level-1);

    auto edgeMatDiff = getEdgeMatrixDiff(move);
    auto edgeCountsDiff = getEdgeCountsDiff(move);

    double weight = 0, degree = 0;
    for (auto neighbor : graph.getNeighboursOfIdx(index)){
        if (index == neighbor.vertexIndex)
            continue;
        auto t = labels[ neighbor.vertexIndex ];
        size_t m = (move.prevLabel >= labelGraph.getSize()) ? 0 : labelGraph.getEdgeMultiplicityIdx(t, move.prevLabel);
        size_t Ert = ((t == move.prevLabel) ? 2 : 1) * (m + edgeMatDiff.get(getOrderedEdge({t, move.prevLabel})));
        size_t Et = edgeCounts.get(t) + edgeCountsDiff.get(t);

        degree += neighbor.label;
        weight += neighbor.label * ( Ert + m_shift ) / (Et + m_shift * (getAvailableLabelCountAtLevel(move.level) + move.addedLabels)) ;
    }


    if (degree == 0)
       return -log(getAvailableLabelCountAtLevel(move.level) + move.addedLabels);
    double logProposal = log(weight) - log(degree);
    return logProposal;
}

template<typename Label>
IntMap<std::pair<Label, Label>> MixedNestedSampler<Label>::getEdgeMatrixDiff(const LabelMove<Label>& move) const {

    Label index = (*m_nestedGraphPriorPtrPtr)->getLabelOfIdx(move.vertexIndex, move.level - 1);
    const auto & labels = (*m_nestedGraphPriorPtrPtr)->getNestedLabels(move.level);
    const auto & graph = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(move.level - 1);

    IntMap<std::pair<Label, Label>> edgeMatDiff;
    Label r = move.prevLabel, s = move.nextLabel;

    for (auto neighbor : graph.getNeighboursOfIdx(index)){
        Label t = labels[neighbor.vertexIndex];
        if (index == neighbor.vertexIndex)
            t = move.prevLabel;
        edgeMatDiff.decrement(getOrderedEdge({r, t}), neighbor.label);

        if (index == neighbor.vertexIndex)
            t = move.nextLabel;
        edgeMatDiff.increment(getOrderedEdge({s, t}), neighbor.label);
    }

    return edgeMatDiff;
}

template<typename Label>
IntMap<Label> MixedNestedSampler<Label>::getEdgeCountsDiff(const LabelMove<Label>& move) const {
    IntMap<Label> edgeCountsDiff;
    Label index = (*m_nestedGraphPriorPtrPtr)->getLabelOfIdx(move.vertexIndex, move.level - 1);
    size_t degree = (*m_nestedGraphPriorPtrPtr)->getNestedLabelGraph(move.level - 1).getDegreeOfIdx(index);
    edgeCountsDiff.decrement(move.prevLabel, degree);
    edgeCountsDiff.increment(move.nextLabel, degree);
     return edgeCountsDiff;
}

template<typename Label>
class GibbsMixedNestedLabelProposer: public GibbsLabelProposer<Label>, public MixedNestedSampler<Label>{
protected:
    const Label sampleLabelUniformlyAtLevel(Level level) const override {
        return std::uniform_int_distribution<size_t>(0, getAvailableLabelCountAtLevel(level) - 2)(rng);
    }
    const size_t getAvailableLabelCountAtLevel(Level level) const override {
        return m_graphPriorPtr->getNestedLabelCount(level);
    }
    const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const override {
        return _getLogProposalProbForReverseMove(move);
    }
    const double getLogProposalProbForMove(const LabelMove<Label>& move) const override {
        return _getLogProposalProbForMove(move);
    }
    using GibbsLabelProposer<Label>::m_graphPriorPtr;
public:
    using GibbsLabelProposer<Label>::sampleLevel;
    using MixedNestedSampler<Label>::_proposeLabelMove;
    using MixedNestedSampler<Label>::_getLogProposalProbForMove;
    using MixedNestedSampler<Label>::_getLogProposalProbForReverseMove;

    GibbsMixedNestedLabelProposer(double sampleLabelCountProb=0.5, double labelCreationProb=0.1, double shift=1):
        GibbsLabelProposer<Label>(sampleLabelCountProb, labelCreationProb),
        MixedNestedSampler<Label>(shift) { this->m_nestedGraphPriorPtrPtr = &this->m_graphPriorPtr; }

    const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        Level level = sampleLevel();
        return _proposeLabelMove(vertex, level);
    }
};

using GibbsMixedNestedBlockProposer = GibbsMixedNestedLabelProposer<BlockIndex>;

template<typename Label>
class RestrictedMixedNestedLabelProposer: public RestrictedNestedLabelProposer<Label>, public MixedNestedSampler<Label>{
protected:
    const Label sampleLabelUniformlyAtLevel(Level level) const override {
        return *sampleUniformlyFrom(m_availableLabels[level].begin(), m_availableLabels[level].end());
    }
    const size_t getAvailableLabelCountAtLevel(Level level) const override {
        return m_availableLabels[level].size();
    }
    const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const override {
        return _getLogProposalProbForReverseMove(move);
    }
    const double getLogProposalProbForMove(const LabelMove<Label>& move) const override {
        return _getLogProposalProbForMove(move);
    }

    using RestrictedNestedLabelProposer<Label>::m_availableLabels;
    using RestrictedNestedLabelProposer<Label>::m_emptyLabels;
    using RestrictedNestedLabelProposer<Label>::m_nestedGraphPriorPtr;
    using MixedNestedSampler<Label>::m_nestedGraphPriorPtrPtr;
public:
    using RestrictedNestedLabelProposer<Label>::sampleLevel;
    using MixedNestedSampler<Label>::_proposeLabelMoveAtLevel;
    using MixedNestedSampler<Label>::_getLogProposalProbForMove;
    using MixedNestedSampler<Label>::_getLogProposalProbForReverseMove;
    RestrictedMixedNestedLabelProposer(double sampleLabelCountProb=0.5, double shift=1):
        RestrictedNestedLabelProposer<Label>(sampleLabelCountProb),
        MixedNestedSampler<Label>(shift) { m_nestedGraphPriorPtrPtr = &m_nestedGraphPriorPtr; }

    const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex&vertex) const override {
        Level level = sampleLevel();
        LabelMove<Label> move = _proposeLabelMoveAtLevel(vertex, level);
        move.addedLabels = -(int) RestrictedNestedLabelProposer<Label>::destroyingLabel(move);
        return move;
    }
};
using RestrictedMixedNestedBlockProposer = RestrictedMixedNestedLabelProposer<BlockIndex>;

} // namespace FastMIDyNet


#endif
