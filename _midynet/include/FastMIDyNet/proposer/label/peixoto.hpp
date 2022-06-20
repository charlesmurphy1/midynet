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
class LabelPeixotoProposer: public LabelProposer<Label> {
private:
    const double m_labelCreationProbability;
    const double m_shift;
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0., 1.);
    mutable std::bernoulli_distribution m_createNewLabelDistribution;
public:
    LabelPeixotoProposer(double createNewLabelProbability=0.1, double shift=1):
        m_createNewLabelDistribution(createNewLabelProbability),
        m_labelCreationProbability(createNewLabelProbability),
        m_shift(shift) {
        assertValidProbability(createNewLabelProbability);
    }
    const LabelMove<Label> proposeMove(const BaseGraph::VertexIndex&) const override ;
    const double getLogProposalProb(const LabelMove<Label>& move) const;
    const double getReverseLogProposalProb(const LabelMove<Label>& move) const;
    const double getLogProposalProbRatio(const LabelMove<Label>& move) const override{
        return getReverseLogProposalProb(move) - getLogProposalProb(move);
    }
    const double getNewLabelProb() const { return m_labelCreationProbability; }
    const double getShift() const { return m_shift; }
protected:
    IntMap<std::pair<Label, Label>> getEdgeMatrixDiff(const LabelMove<Label>& move) const ;
    IntMap<Label> getEdgeCountsDiff(const LabelMove<Label>& move) const ;
};

class BlockPeixotoProposer: public LabelPeixotoProposer<BlockIndex> {
public:
    using LabelPeixotoProposer<BlockIndex>::LabelPeixotoProposer;
};


template<typename Label>
const LabelMove<Label> LabelPeixotoProposer<Label>::proposeMove(const BaseGraph::VertexIndex& movedVertex) const {

    const auto & labels = *this->m_labelsPtr;
    const auto & graph = *this->m_graphPtr, labelGraph = *this->m_labelGraphPtr;
    const auto & edgeCounts = *this->m_edgeLabelCountsPtr;

    Label prevLabel = labels[movedVertex], nextLabel;
    size_t B = this->m_labelCountsPtr->size();
    if (m_createNewLabelDistribution(rng) == 1)
        return {movedVertex, prevLabel, B};
    if ( graph.getDegreeOfIdx(movedVertex) == 0 ){
        std::uniform_int_distribution<size_t> dist(0, B-1);
        Label nextLabel = dist(rng);
        LabelMove<Label> move = {movedVertex, prevLabel, nextLabel};
        return move;
    }

    auto neighbors = graph.getNeighboursOfIdx(movedVertex);
    BaseGraph::VertexIndex randomNeighbor = movedVertex;
    while(randomNeighbor == movedVertex)
        randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
    Label t = labels[randomNeighbor];

    double probUniformSampling = m_shift * B / (edgeCounts.get(t) + m_shift * B);
    if ( m_uniform01(rng) < probUniformSampling){
        std::uniform_int_distribution<size_t> dist(0, B-1);
        nextLabel = dist(rng);
    } else {
        std::uniform_int_distribution<int> dist(0, edgeCounts.get(t) - 1);
        int mult = dist(rng);
        for (auto s : labelGraph.getNeighboursOfIdx(t)){
            mult -= ((t == s.vertexIndex) ? 2 : 1) * s.label;
            nextLabel = s.vertexIndex;
            if (mult < 0) break;
        }
    }

    LabelMove<Label> move = {movedVertex, prevLabel, nextLabel};
    return move;
}

template<typename Label>
const double LabelPeixotoProposer<Label>::getLogProposalProb(const LabelMove<Label>& move) const {

    const auto & labels = *this->m_labelsPtr;
    const auto & graph = *this->m_graphPtr, labelGraph = *this->m_labelGraphPtr;
    const auto & edgeCounts = *this->m_edgeLabelCountsPtr;

    size_t B = this->m_labelCountsPtr->size();
    if ( this->creatingNewLabel(move) )
         return log(m_labelCreationProbability);
    double weight = 0, degree = 0;
    for (auto neighbor : graph.getNeighboursOfIdx(move.vertexIndex)){
        if (move.vertexIndex == neighbor.vertexIndex)
            continue;
        auto t = labels[ neighbor.vertexIndex ];
        size_t Est = ((t == move.nextLabel) ? 2 : 1) * labelGraph.getEdgeMultiplicityIdx(t, move.nextLabel);
        size_t Et = edgeCounts.get(t);

        degree += neighbor.label;
        weight += neighbor.label * ( Est + m_shift ) / (Et + m_shift * B) ;
    }

    if (degree == 0)
       return log(1 - m_labelCreationProbability) - log(B);
    double logProposal = log(1 - m_labelCreationProbability) + log(weight) - log(degree);
    return logProposal;
}


template<typename Label>
IntMap<std::pair<Label, Label>> LabelPeixotoProposer<Label>::getEdgeMatrixDiff(const LabelMove<Label>& move) const {

    const auto & labels = *this->m_labelsPtr;
    const auto & graph = *this->m_graphPtr;

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
IntMap<Label> LabelPeixotoProposer<Label>::getEdgeCountsDiff(const LabelMove<Label>& move) const {
    IntMap<Label> edgeCountsDiff;

    size_t degree = this->m_graphPtr->getDegreeOfIdx(move.vertexIndex);
    edgeCountsDiff.decrement(move.prevLabel, degree);
    edgeCountsDiff.increment(move.nextLabel, degree);
     return edgeCountsDiff;
}

template<typename Label>
const double LabelPeixotoProposer<Label>::getReverseLogProposalProb(const LabelMove<Label>& move) const {

    const auto & labels = *this->m_labelsPtr;
    const auto & graph = *this->m_graphPtr, labelGraph = *this->m_labelGraphPtr;
    const auto & edgeCounts = *this->m_edgeLabelCountsPtr;

    int addedBlocks = this->getAddedLabels(move) ;
    size_t B = this->m_labelCountsPtr->size() + addedBlocks;
    if ( this->destroyingLabel(move) )
         return log(m_labelCreationProbability);

    auto edgeMatDiff = getEdgeMatrixDiff(move);
    auto edgeCountsDiff = getEdgeCountsDiff(move);


    double weight = 0, degree = 0;
    for (auto neighbor : graph.getNeighboursOfIdx(move.vertexIndex)){
        if (move.vertexIndex == neighbor.vertexIndex)
            continue;
        auto t = labels[ neighbor.vertexIndex ];
        size_t Ert = ((t == move.prevLabel) ? 2 : 1) * (labelGraph.getEdgeMultiplicityIdx(t, move.prevLabel) + edgeMatDiff.get({t, move.prevLabel}));
        size_t Et = edgeCounts.get(t) + edgeCountsDiff.get(t);
        degree += neighbor.label;
        weight += neighbor.label * ( Ert + m_shift ) / (Et + m_shift * B) ;
    }

    if (degree == 0)
       return log(1 - m_labelCreationProbability) - log(B);
    double logReverseProposal = log(1 - m_labelCreationProbability) + log ( weight ) - log( degree );
    return logReverseProposal;
}


} // namespace FastMIDyNet


#endif
