#ifndef FAST_MIDYNET_PEIXOTO_PROPOSER_H
#define FAST_MIDYNET_PEIXOTO_PROPOSER_H


#include "SamplableSet.hpp"

#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"

namespace FastMIDyNet {

template<typename Label>
class MixedSampler{
protected:
    double m_shift;
    const VertexLabeledRandomGraph<Label>** m_graphPriorPtrPtr = nullptr;
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);

    Label sampleNeighborLabel(BaseGraph::VertexIndex vertex) const {
        size_t degree = (*m_graphPriorPtrPtr)->getGraph().getDegreeOfIdx(vertex);
        size_t counter = std::uniform_int_distribution<size_t>(0, degree-1)(rng);
        for (const auto& neighbor : (*m_graphPriorPtrPtr)->getGraph().getNeighboursOfIdx(vertex)){
            counter -= neighbor.label;
            if (counter < 0)
                return (*m_graphPriorPtrPtr)->getLabelOfIdx(neighbor.vertexIndex);
        }
        return  (*m_graphPriorPtrPtr)->getLabelOfIdx(vertex);
    }

    virtual const Label sampleLabelUniformly() const = 0;

    const Label sampleLabelPreferentially(const Label neighborLabel) const ;
    const double _getLogProposalProbForMove(const LabelMove<Label>& move) const ;
    const double _getLogProposalProbForReverseMove(const LabelMove<Label>& move) const ;
    const LabelMove<Label> _proposeLabelMove(const BaseGraph::VertexIndex&) const ;
    IntMap<std::pair<Label, Label>> getEdgeMatrixDiff(const LabelMove<Label>& move) const ;
    IntMap<Label> getEdgeCountsDiff(const LabelMove<Label>& move) const ;
    virtual const size_t getAvailableLabelCount() const = 0;

    bool creatingNewLabel(const LabelMove<Label>& move) const {
        return (*m_graphPriorPtrPtr)->getLabelCounts().get(move.nextLabel) == 0;
    };
    bool destroyingLabel(const LabelMove<Label>& move) const {
        return move.prevLabel != move.nextLabel and (*m_graphPriorPtrPtr)->getLabelCounts().get(move.prevLabel) == 1 ;
    }
 public:
     MixedSampler(double shift=1): m_shift(shift) {}

    const double getShift() const { return m_shift; }
};

template<typename Label>
const Label MixedSampler<Label>::sampleLabelPreferentially(const Label neighborLabel) const {
    std::uniform_int_distribution<int> dist(0, (*m_graphPriorPtrPtr)->getEdgeLabelCounts().get(neighborLabel) - 1);
    int mult = dist(rng);
    for (auto s : (*m_graphPriorPtrPtr)->getLabelGraph().getNeighboursOfIdx(neighborLabel)){
        mult -= ((neighborLabel == s.vertexIndex) ? 2 : 1) * s.label;
        if (mult < 0)
            return s.vertexIndex;
    }
    return sampleLabelUniformly();
}

template<typename Label>
const LabelMove<Label> MixedSampler<Label>::_proposeLabelMove(const BaseGraph::VertexIndex&vertex) const {
    const auto& edgeCounts = (*m_graphPriorPtrPtr)->getEdgeLabelCounts();
    const auto& B = getAvailableLabelCount();
    Label neighborLabel = MixedSampler<Label>::sampleNeighborLabel(vertex);
    double probUniformSampling = m_shift * B / (edgeCounts.get(neighborLabel) + m_shift * B);
    Label nextLabel = (m_uniform01(rng) < probUniformSampling) ? sampleLabelUniformly() : sampleLabelPreferentially(neighborLabel);
    return {vertex, (*m_graphPriorPtrPtr)->getLabelOfIdx(vertex), nextLabel};
}

template<typename Label>
const double MixedSampler<Label>::_getLogProposalProbForMove(const LabelMove<Label>& move) const {
    const auto & labels = (*m_graphPriorPtrPtr)->getLabels();
    const auto & edgeCounts = (*m_graphPriorPtrPtr)->getEdgeLabelCounts();
    const auto & graph = (*m_graphPriorPtrPtr)->getGraph();
    const auto &labelGraph = (*m_graphPriorPtrPtr)->getLabelGraph();

    double weight = 0, degree = 0;
    for (auto neighbor : graph.getNeighboursOfIdx(move.vertexIndex)){
        if (move.vertexIndex == neighbor.vertexIndex)
            continue;
        auto t = labels[ neighbor.vertexIndex ];
        size_t m = (move.nextLabel < labelGraph.getSize()) ? labelGraph.getEdgeMultiplicityIdx(t, move.nextLabel) : 0;
        size_t Est = ((t == move.nextLabel) ? 2 : 1) * m;
        size_t Et = edgeCounts.get(t);

        degree += neighbor.label;
        weight += neighbor.label * ( Est + m_shift ) / (Et + m_shift * getAvailableLabelCount()) ;
    }

    if (degree == 0)
       return -log(getAvailableLabelCount());
    double logProposal = log(weight) - log(degree);
    return logProposal;
}

template<typename Label>
const double MixedSampler<Label>::_getLogProposalProbForReverseMove(const LabelMove<Label>& move) const {
    const auto & labels = (*m_graphPriorPtrPtr)->getLabels();
    const auto & edgeCounts = (*m_graphPriorPtrPtr)->getEdgeLabelCounts();
    const auto & graph = (*m_graphPriorPtrPtr)->getGraph();
    const auto &labelGraph = (*m_graphPriorPtrPtr)->getLabelGraph();

    auto edgeMatDiff = getEdgeMatrixDiff(move);
    auto edgeCountsDiff = getEdgeCountsDiff(move);

    double weight = 0, degree = 0;
    for (auto neighbor : graph.getNeighboursOfIdx(move.vertexIndex)){
        if (move.vertexIndex == neighbor.vertexIndex)
            continue;
        auto t = labels[ neighbor.vertexIndex ];
        size_t m = (move.prevLabel < labelGraph.getSize()) ? labelGraph.getEdgeMultiplicityIdx(t, move.prevLabel) : 0;
        size_t Ert = ((t == move.prevLabel) ? 2 : 1) * ((m + edgeMatDiff.get(getOrderedEdge({t, move.prevLabel}))));
        size_t Et = edgeCounts.get(t) + edgeCountsDiff.get(t);
        degree += neighbor.label;
        weight += neighbor.label * ( Ert + m_shift ) / (Et + m_shift * getAvailableLabelCount()) ;
    }

    if (degree == 0)
       return -log(getAvailableLabelCount());
    double logProposal = log(weight) - log(degree);
    return logProposal;
}

template<typename Label>
IntMap<std::pair<Label, Label>> MixedSampler<Label>::getEdgeMatrixDiff(const LabelMove<Label>& move) const {

    const auto & labels = (*m_graphPriorPtrPtr)->getLabels();
    const auto & graph = (*m_graphPriorPtrPtr)->getGraph();

    IntMap<std::pair<Label, Label>> edgeMatDiff;
    Label r = move.prevLabel, s = move.nextLabel;
    for (auto neighbor : graph.getNeighboursOfIdx(move.vertexIndex)){
        Label t = labels[neighbor.vertexIndex];
        if (move.vertexIndex == neighbor.vertexIndex)
            t = move.prevLabel;
        edgeMatDiff.decrement(getOrderedEdge({r, t}), neighbor.label);
        if (move.vertexIndex == neighbor.vertexIndex)
            t = move.nextLabel;
        edgeMatDiff.increment(getOrderedEdge({s, t}), neighbor.label);
    }
     return edgeMatDiff;
}

template<typename Label>
IntMap<Label> MixedSampler<Label>::getEdgeCountsDiff(const LabelMove<Label>& move) const {
    IntMap<Label> edgeCountsDiff;
    size_t degree = (*m_graphPriorPtrPtr)->getGraph().getDegreeOfIdx(move.vertexIndex);
    edgeCountsDiff.decrement(move.prevLabel, degree);
    edgeCountsDiff.increment(move.nextLabel, degree);
     return edgeCountsDiff;
}

template<typename Label>
class GibbsMixedLabelProposer: public GibbsLabelProposer<Label>, public MixedSampler<Label>{
protected:
    const Label sampleLabelUniformly() const override {
        return std::uniform_int_distribution<size_t>(0, getAvailableLabelCount() - 2)(rng);
    }
    const size_t getAvailableLabelCount() const override { return GibbsLabelProposer<Label>::m_graphPriorPtr->getLabelCount(); }
    const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const override {
        return MixedSampler<Label>::_getLogProposalProbForReverseMove(move);
    }
    const double getLogProposalProbForMove(const LabelMove<Label>& move) const override {
        return MixedSampler<Label>::_getLogProposalProbForMove(move);
    }
public:
    GibbsMixedLabelProposer(double sampleLabelCountProb=0.5, double labelCreationProb=0.1, double shift=1):
        GibbsLabelProposer<Label>(sampleLabelCountProb, labelCreationProb),
        MixedSampler<Label>(shift) { this->m_graphPriorPtrPtr = &this->m_graphPriorPtr; }

    const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex&vertex) const override {
        return MixedSampler<Label>::_proposeLabelMove(vertex);
    }
};

using GibbsMixedBlockProposer = GibbsMixedLabelProposer<BlockIndex>;

template<typename Label>
class RestrictedMixedLabelProposer: public RestrictedLabelProposer<Label>, public MixedSampler<Label>{
protected:
    const Label sampleLabelUniformly() const override { return *sampleUniformlyFrom(m_availableLabels.begin(), m_availableLabels.end()); }
    const size_t getAvailableLabelCount() const override { return m_availableLabels.size(); }
    const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const override {
        return MixedSampler<Label>::_getLogProposalProbForReverseMove(move);
    }
    const double getLogProposalProbForMove(const LabelMove<Label>& move) const override {
        return MixedSampler<Label>::_getLogProposalProbForMove(move);
    }

    using RestrictedLabelProposer<Label>::m_availableLabels;
    using RestrictedLabelProposer<Label>::m_emptyLabels;
public:
    RestrictedMixedLabelProposer(double sampleLabelCountProb=0.5, double shift=1):
        RestrictedLabelProposer<Label>(sampleLabelCountProb),
        MixedSampler<Label>(shift) { this->m_graphPriorPtrPtr = &this->m_graphPriorPtr; }

    const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex&vertex) const override {
        return MixedSampler<Label>::_proposeLabelMove(vertex);
    }
};
using RestrictedMixedBlockProposer = RestrictedMixedLabelProposer<BlockIndex>;

} // namespace FastMIDyNet


#endif
