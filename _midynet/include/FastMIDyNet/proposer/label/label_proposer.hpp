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
    const VertexLabeledRandomGraph<Label>* m_graphPriorPtr = nullptr;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;
    const double m_sampleLabelCountProb;
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);
    Label m_nextNewLabel;

    virtual void initNewLabel() = 0;
public:
    LabelProposer(double sampleLabelCountProb=0.1):
        m_sampleLabelCountProb(sampleLabelCountProb) { }
    void setUp(const VertexLabeledRandomGraph<Label>& randomGraph) {
        m_graphPriorPtr = &randomGraph;
        m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
        initNewLabel();
    }
    virtual const double getLogProposalProbRatio(const MoveType& move) const = 0;
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyLabelMove(const MoveType& move) = 0;
    const double getSampleLabelCountProb() const { return m_sampleLabelCountProb; }
    virtual const MoveType proposeLabelMove(const BaseGraph::VertexIndex&) const = 0;

    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const {
        return {vertex, m_graphPriorPtr->getLabelOfIdx(vertex), m_nextNewLabel, 1};
    }

    const MoveType proposeMove() const {
        BaseGraph::VertexIndex vertex = m_vertexDistribution(rng);
        if (m_uniform01(rng) < m_sampleLabelCountProb)
            return proposeNewLabelMove(vertex);
        return proposeLabelMove(vertex);
    }
    void checkSelfSafety() const override{
        if (m_graphPriorPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_graphPriorPtr` is NULL.");
    }
};

template<typename Label>
class UnrestrictedLabelProposer: public LabelProposer<Label>{
protected:
    double m_labelCreationProb;
    using BaseClass = LabelProposer<Label>;
    using BaseClass::m_nextNewLabel;
    using BaseClass::m_graphPriorPtr;
    void initNewLabel() override { m_nextNewLabel = m_graphPriorPtr->getLabelCount();}
public:
    UnrestrictedLabelProposer(double labelCreationProb=0.5, double sampleLabelCountProb=0.1):
        LabelProposer<Label>(sampleLabelCountProb), m_labelCreationProb(labelCreationProb) {}
    virtual void applyLabelMove(const LabelMove<Label>& move) override { initNewLabel(); }
};

template<typename Label>
class RestrictedLabelProposer: public LabelProposer<Label>{
protected:
    std::set<Label> m_emptyLabels, m_availableLabels;
    void initNewLabel() override { m_nextNewLabel = m_labelCountsPtr->getSize();}
    bool creatingNewLabel(const LabelMove<Label>& move) const {
        return m_labelCountsPtr->get(move.nextLabel) == 0;
    };
    bool destroyingLabel(const LabelMove<Label>& move) const {
        return move.prevLabel != move.nextLabel and m_labelCountsPtr->get(move.prevLabel) == 1 ;
    }
    const int getAddedLabels(const LabelMove<Label>& move) const {
        return (int) creatingNewLabel(move) - (int) destroyingLabel(move);
    }
    using BaseClass = LabelProposer<Label>;
    using BaseClass::m_labelsPtr;
    using BaseClass::m_labelCountsPtr;
    using BaseClass::m_nextNewLabel;
public:
    using LabelProposer<Label>::LabelProposer;

    virtual void applyLabelMove(const LabelMove<Label>& move) override {
        if ( destroyingLabel(move) )
            m_emptyLabels.insert(move.prevLabel);
        if ( creatingNewLabel(move) ){
            m_availableLabels.insert(move.nextLabel);
            if (m_emptyLabels.size() == 0)
                ++m_nextNewLabel;
            else
                m_nextNewLabel = sampleUniformlyFrom(m_emptyLabels.begin(), m_emptyLabels.end());
        }
    }


};

} // namespace FastMIDyNet


#endif
