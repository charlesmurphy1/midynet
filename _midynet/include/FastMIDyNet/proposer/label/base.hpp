#ifndef FAST_MIDYNET_LABELPROPOSER_H
#define FAST_MIDYNET_LABELPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet {

template<typename Label>
class LabelProposer: public Proposer<LabelMove<Label>> {
protected:
    const VertexLabeledRandomGraph<Label>* m_graphPriorPtr = nullptr;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);
    const double m_sampleLabelCountProb;

public:
    LabelProposer(double sampleLabelCountProb=0.1):
        m_sampleLabelCountProb(sampleLabelCountProb) { }
    virtual void setUp(const VertexLabeledRandomGraph<Label>& graphPrior) {
        m_graphPriorPtr = &graphPrior;
        m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, graphPrior.getSize() - 1);
    }
    const double getLogProposalProbRatio(const LabelMove<Label>& move) const {
        return getLogProposalProb(move, true) - getLogProposalProb(move);
    }
    virtual const double getLogProposalProb(const LabelMove<Label>& move, bool reverse=false) const = 0;
    const double getSampleLabelCountProb() const { return m_sampleLabelCountProb; }
    virtual void applyLabelMove(const LabelMove<Label>& move) { };


    const LabelMove<Label> proposeMove() const {
        BaseGraph::VertexIndex vertex = m_vertexDistribution(rng);
        if (m_uniform01(rng) < m_sampleLabelCountProb)
            return proposeNewLabelMove(vertex);
        return proposeLabelMove(vertex);
    }
    virtual const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex&) const = 0;
    virtual const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex&) const = 0;
    void checkSelfSafety() const override{
        if (m_graphPriorPtr == nullptr)
            throw SafetyError("LabelProposer: unsafe proposer since `m_graphPriorPtr` is NULL.");
    }
};

template<typename Label>
class GibbsLabelProposer: public LabelProposer<Label>{
protected:
    double m_labelCreationProb;
    using BaseClass = LabelProposer<Label>;
    using BaseClass::m_graphPriorPtr;
    using BaseClass::m_uniform01;
    using BaseClass::m_vertexDistribution;
    using BaseClass::m_sampleLabelCountProb;
    virtual const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const = 0;
    virtual const double getLogProposalProbForMove(const LabelMove<Label>& move) const = 0;

public:
    GibbsLabelProposer(double sampleLabelCountProb=0.1, double labelCreationProb=0.5):
        LabelProposer<Label>(sampleLabelCountProb), m_labelCreationProb(labelCreationProb) {}
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        if ( m_uniform01(rng) < m_labelCreationProb )
            return {vertex, m_graphPriorPtr->getLabelOfIdx(vertex), m_graphPriorPtr->getLabelCount(), 1};
        else
            return {vertex, m_graphPriorPtr->getLabelOfIdx(vertex), m_graphPriorPtr->getLabelOfIdx(vertex), -1};
    }

    const double getLogProposalProb(const LabelMove<Label>& move, bool reverse=false) const override {
        if ( move.addedLabels != 0)
            return log(m_sampleLabelCountProb);
        return log(1 - m_sampleLabelCountProb) + ((reverse) ? getLogProposalProbForReverseMove(move) :  getLogProposalProbForMove(move));
    }
};

template<typename Label>
class RestrictedLabelProposer: public LabelProposer<Label>{
protected:
    std::set<Label> m_emptyLabels, m_availableLabels;
    bool creatingNewLabel(const LabelMove<Label>& move) const {
        return m_graphPriorPtr->getLabelCounts().get(move.nextLabel) == 0;
    };
    bool destroyingLabel(const LabelMove<Label>& move) const {
        return move.prevLabel != move.nextLabel and m_graphPriorPtr->getLabelCounts().get(move.prevLabel) == 1 ;
    }
    int getAddedLabels(const LabelMove<Label>& move) const {
        return (int) creatingNewLabel(move) - (int) destroyingLabel(move);
    }
    using BaseClass = LabelProposer<Label>;
    using BaseClass::m_graphPriorPtr;
    using BaseClass::m_uniform01;
    using BaseClass::m_vertexDistribution;
    using BaseClass::m_sampleLabelCountProb;
    virtual const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const = 0;
    virtual const double getLogProposalProbForMove(const LabelMove<Label>& move) const = 0;

public:
    using LabelProposer<Label>::LabelProposer;
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        Label prevLabel = m_graphPriorPtr->getLabelOfIdx(vertex);
        Label nextLabel = *sampleUniformlyFrom(m_emptyLabels.begin(), m_emptyLabels.end());
        LabelMove<Label> move = {vertex, prevLabel, nextLabel};
        if ( destroyingLabel(move) )
            return {vertex, prevLabel, prevLabel};
        move.addedLabels = 1;
        return move;
    }
    void setUp(const VertexLabeledRandomGraph<Label>& graphPrior) override {
        BaseClass::setUp(graphPrior);
        m_emptyLabels.clear();
        m_availableLabels.clear();
        m_emptyLabels.insert(m_graphPriorPtr->getLabelCount());
        for (const auto& nr: graphPrior.getLabelCounts())
            m_availableLabels.insert(nr.first);
    }

    const double getLogProposalProb(const LabelMove<Label>& move, bool reverse=false) const override {
        if ( move.addedLabels == ((reverse) ? -1 : 1 ))
            return log(m_sampleLabelCountProb);
        return log(1 - m_sampleLabelCountProb) + ((reverse) ? getLogProposalProbForReverseMove(move) :  getLogProposalProbForMove(move));
    }

    void applyLabelMove(const LabelMove<Label>& move) override {
        if ( destroyingLabel(move) and move.prevLabel != move.nextLabel ){
            m_emptyLabels.insert(move.prevLabel);
            m_availableLabels.erase(move.prevLabel);
        }
        if ( creatingNewLabel(move) ){
            m_availableLabels.insert(move.nextLabel);
            m_emptyLabels.erase(move.nextLabel);
        }
        if (m_emptyLabels.size() == 0)
            m_emptyLabels.insert(m_graphPriorPtr->getLabelCount());
    }


};

} // namespace FastMIDyNet


#endif
