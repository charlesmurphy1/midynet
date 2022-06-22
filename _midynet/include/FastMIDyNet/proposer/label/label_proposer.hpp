#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet {

template<typename Label>
class LabelProposer: public Proposer<LabelMove<Label>> {
protected:
    using MoveType = LabelMove<Label>;
    const std::vector<Label>* m_labelsPtr = nullptr;
    const CounterMap<Label>* m_labelCountsPtr = nullptr;
    const CounterMap<Label>* m_edgeLabelCountsPtr = nullptr;
    const MultiGraph* m_labelGraphPtr = nullptr;
    const MultiGraph* m_graphPtr = nullptr;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;

    bool creatingNewLabel(const MoveType& move) const {
        return m_labelCountsPtr->get(move.nextLabel) == 0;
    };
    bool destroyingLabel(const MoveType& move) const {
        return move.prevLabel != move.nextLabel and m_labelCountsPtr->get(move.prevLabel) == 1 ;
    }
    const int getAddedLabels(const MoveType& move) const {
        return (int) creatingNewLabel(move) - (int) destroyingLabel(move);
    }
public:
    void setUp(const VertexLabeledRandomGraph<Label>& randomGraph) {
        m_labelsPtr = &randomGraph.getVertexLabels();
        m_labelCountsPtr = &randomGraph.getLabelCounts();
        m_edgeLabelCountsPtr = &randomGraph.getEdgeLabelCounts();
        m_graphPtr = &randomGraph.getGraph();
        m_labelGraphPtr = &randomGraph.getLabelGraph();
        m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
    };
    virtual const double getLogProposalProbRatio(const MoveType& move) const = 0;
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyLabelMove(const MoveType& move) {};
    virtual const MoveType proposeMove(const BaseGraph::VertexIndex&) const = 0;
    const MoveType proposeMove() const override{
        auto vertexIdx = m_vertexDistribution(rng);
        return proposeMove(vertexIdx);
    }
    void checkSelfSafety() const override{
        if (m_labelsPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_labelsPtr` is NULL.");
        if (m_labelCountsPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_labelCountsPtr` is NULL.");
        if (m_edgeLabelCountsPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_edgeLabelCountsPtr` is NULL.");
        if (m_graphPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_graphPtr` is NULL.");
        if (m_labelGraphPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_labelGraphPtr` is NULL.");
    }
};

} // namespace FastMIDyNet


#endif
