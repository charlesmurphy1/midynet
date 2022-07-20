#ifndef FAST_MIDYNET_NESTED_LABELPROPOSER_H
#define FAST_MIDYNET_NESTED_LABELPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet {

template<typename Label>
class NestedLabelProposer: public LabelProposer<Label> {
protected:
    const NestedVertexLabeledRandomGraph<Label>* m_nestedGraphPriorPtr = nullptr;
public:
    NestedLabelProposer(double sampleLabelCountProb=0.1):
        LabelProposer<Label>(sampleLabelCountProb) { }
    virtual void setUp(const NestedVertexLabeledRandomGraph<Label>& graphPrior) override {
        LabelProposer<Label>::setUp(graphPrior);
        m_nestedGraphPriorPtr = &graphPrior;
    }
    const Level sampleLevel() {
        std::uniform_int_distribution<Level> dist(0, m_nestedGraphPriorPtr->getDepth() - 1);
        return dist(rng);
    }
    void checkSelfSafety() const override{
        LabelProposer<Label>::checkSelfSafety();
        if (m_nestedGraphPriorPtr == nullptr)
            throw SafetyError("NestedLabelProposer: unsafe proposer since `m_nestedGraphPriorPtr` is `nullptr`.");
    }
};

template<typename Label>
class GibbsNestedLabelProposer: public NestedLabelProposer<Label>{
protected:
    double m_labelCreationProb;
    using BaseClass = LabelProposer<Label>;
    using NestedBaseClass = NestedLabelProposer<Label>;
    using NestedBaseClass::m_nestedGraphPriorPtr;
    using BaseClass::m_uniform01;
    using BaseClass::m_vertexDistribution;
    using BaseClass::m_sampleLabelCountProb;
    virtual const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const = 0;
    virtual const double getLogProposalProbForMove(const LabelMove<Label>& move) const = 0;
    bool creatingNewLevel(const LabelMove<Label>& move) const {
        return move.addedLabels==1 and move.level==m_nestedGraphPriorPtr->getDepth()-1;
    }
    bool destroyingLevel(const LabelMove<Label>& move) const {
        return move.addedLabels==-1 and move.level==m_nestedGraphPriorPtr->getDepth()-1;
    }

public:
    GibbsNestedLabelProposer(double sampleLabelCountProb=0.1, double labelCreationProb=0.5):
        NestedLabelProposer<Label>(sampleLabelCountProb), m_labelCreationProb(labelCreationProb) {}
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        Level level = NestedBaseClass::sampleLevel(vertex);

        if ( m_uniform01(rng) < m_labelCreationProb )
            return {vertex, m_nestedGraphPriorPtr->getNestedLabelOfIdx(vertex, level), m_nestedGraphPriorPtr->getNestedLabelCount()[level], 1, level};
        else
            return {vertex, m_nestedGraphPriorPtr->getNestedLabelOfIdx(vertex, level), m_nestedGraphPriorPtr->getNestedLabelOfIdx(vertex, level), -1, level};
    }

    const double getLogProposalProb(const LabelMove<Label>& move, bool reverse=false) const override {
        if ( move.addedLabels != 0){
            int dL = (reverse) ? (creatingNewLevel(move) ? -1 : 1) : 0;
            return log(m_sampleLabelCountProb) - log(m_nestedGraphPriorPtr->getDepth() + dL);
        }
        return log(1 - m_sampleLabelCountProb) - log(m_nestedGraphPriorPtr->getDepth()) + ((reverse) ? getLogProposalProbForReverseMove(move) :  getLogProposalProbForMove(move));
    }
};

template<typename Label>
class RestrictedNestedLabelProposer: public NestedLabelProposer<Label>{
protected:
    std::vector<std::set<Label>> m_emptyLabels, m_availableLabels;
    bool creatingNewLevel(const LabelMove<Label>& move) const {
        return creatingNewLabel(move) and move.level==m_nestedGraphPriorPtr->getDepth()-1;
    }
    bool destroyingLevel(const LabelMove<Label>& move) const {
        return destroyingLabel(move) and move.level==m_nestedGraphPriorPtr->getDepth()-1;
    }

    bool creatingNewLabel(const LabelMove<Label>& move) const {
        return m_nestedGraphPriorPtr->getNestedLabelCounts()[move.level].get(move.nextLabel) == 0;
    };
    bool destroyingLabel(const LabelMove<Label>& move) const {
        return move.prevLabel != move.nextLabel and m_nestedGraphPriorPtr->getNestedLabelCounts()[move.level].get(move.prevLabel) == 1 ;
    }

    int getAddedLabels(const LabelMove<Label>& move) const {
        return (int) creatingNewLabel(move) - (int) destroyingLabel(move);
    }
    using BaseClass = LabelProposer<Label>;
    using NestedBaseClass = NestedLabelProposer<Label>;
    using NestedBaseClass::m_nestedGraphPriorPtr;
    using NestedBaseClass::sampleLevel;
    using BaseClass::m_uniform01;
    using BaseClass::m_vertexDistribution;
    using BaseClass::m_sampleLabelCountProb;
    virtual const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const = 0;
    virtual const double getLogProposalProbForMove(const LabelMove<Label>& move) const = 0;
//
public:
    using NestedLabelProposer<Label>::NestedLabelProposer;
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        Level level = sampleLevel(vertex);
        Label prevLabel = m_nestedGraphPriorPtr->getNestedLabelOfIdx(vertex, level);
        Label nextLabel = *sampleUniformlyFrom(m_emptyLabels[level].begin(), m_emptyLabels[level].end());
        LabelMove<Label> move = {vertex, prevLabel, nextLabel, level};
        if ( destroyingLabel(move) )
            return {vertex, prevLabel, prevLabel, level};
        move.addedLabels = 1;
        return move;
    }
    void setUp(const NestedVertexLabeledRandomGraph<Label>& graphPrior) override {
        BaseClass::setUp(graphPrior);
        m_emptyLabels.clear();
        m_availableLabels.clear();
        for (Level l=0; l<m_nestedGraphPriorPtr->getDepth(); ++l){
            m_emptyLabels.push_back({m_nestedGraphPriorPtr->getNestedLabelCount()[l]});
            m_availableLabels.push_back({});
            for (const auto& nr: graphPrior.getNestedLabelCounts()[l])
                m_availableLabels[l].insert(nr.first);
        }
    }

    const double getLogProposalProb(const LabelMove<Label>& move, bool reverse=false) const override {
        if ( move.addedLabels == ((reverse) ? -1 : 1 )){
            return log(m_sampleLabelCountProb) - log(m_nestedGraphPriorPtr->getDepth() + (destroyingLevel(move)) ? 1 : 0);
        }
        return log(1 - m_sampleLabelCountProb) - log(m_nestedGraphPriorPtr->getDepth()) + ((reverse) ? getLogProposalProbForReverseMove(move) :  getLogProposalProbForMove(move));
    }

    void applyLabelMove(const LabelMove<Label>& move) override {
        if ( destroyingLabel(move) and move.prevLabel != move.nextLabel ){
            m_emptyLabels[move.level].insert(move.prevLabel);
            m_availableLabels[move.level].erase(move.prevLabel);
        }
        if ( creatingNewLabel(move) ){
            m_availableLabels[move.level].insert(move.nextLabel);
            m_emptyLabels[move.level].erase(move.nextLabel);
        }
        if (m_emptyLabels[move.level].size() == 0)
            m_emptyLabels[move.level].insert(m_nestedGraphPriorPtr->getNestedLabelCount()[move.level]);

        if (creatingNewLevel(move)){
            m_emptyLabels.push_back({1});
            m_availableLabels.push_back({0});
        }
        if (destroyingLevel(move)){
            m_emptyLabels.pop_back();
            m_availableLabels.pop_back();
        }
    }


};

} // namespace FastMIDyNet


#endif
