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
    using LabelProposer<Label>::m_sampleLabelCountProb;
    using LabelProposer<Label>::isCreatingLabelMove;
    using LabelProposer<Label>::getLogProposalProbForReverseMove;
    using LabelProposer<Label>::getLogProposalProbForMove;
public:
    NestedLabelProposer(double sampleLabelCountProb=0.1):
        LabelProposer<Label>(sampleLabelCountProb) { }
    virtual void setUpWithNestedPrior(const NestedVertexLabeledRandomGraph<Label>& graphPrior) {
        LabelProposer<Label>::setUpWithPrior(graphPrior);
        m_nestedGraphPriorPtr = &graphPrior;
    }
    const Level sampleLevel() const {
        std::uniform_int_distribution<Level> dist(0, m_nestedGraphPriorPtr->getDepth() - 1);
        return dist(rng);
    }

    const double getLogProposalProb(const LabelMove<Label>& move, bool reverse=false) const override {
        int dL;
        if ( isCreatingLabelMove(move, reverse) ){
            dL = (reverse and move.level == m_nestedGraphPriorPtr->getDepth() - 2 and m_nestedGraphPriorPtr->getNestedVertexCounts(move.level).size() == 2) ? -1 : 0;
            return log(m_sampleLabelCountProb) - log(m_nestedGraphPriorPtr->getDepth() + dL);
        }
        dL = (reverse and move.level == m_nestedGraphPriorPtr->getDepth() - 1 and move.addedLabels == 1) ? 1 : 0;
        return log(1 - m_sampleLabelCountProb) - log(m_nestedGraphPriorPtr->getDepth() + dL) + ((reverse) ? getLogProposalProbForReverseMove(move) : getLogProposalProbForMove(move));
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
    bool creatingNewLevel(const LabelMove<Label>& move) const {
        return move.addedLabels==1 and move.level==m_nestedGraphPriorPtr->getDepth()-1;
    }
    bool destroyingLevel(const LabelMove<Label>& move) const {
        return move.addedLabels==-1 and move.level==m_nestedGraphPriorPtr->getDepth()-1;
    }
    bool isCreatingLabelMove(const LabelMove<Label>& move, bool reverse) const override {
        return move.addedLabels != 0;
    }

public:
    GibbsNestedLabelProposer(double sampleLabelCountProb=0.1, double labelCreationProb=0.5):
        NestedLabelProposer<Label>(sampleLabelCountProb), m_labelCreationProb(labelCreationProb) {}
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        Level level = NestedBaseClass::sampleLevel();
        if ( m_uniform01(rng) < m_labelCreationProb )
            return {vertex, m_nestedGraphPriorPtr->getLabelOfIdx(vertex, level), m_nestedGraphPriorPtr->getNestedLabelCount()[level], 1, level};
        else
            return {vertex, m_nestedGraphPriorPtr->getLabelOfIdx(vertex, level), m_nestedGraphPriorPtr->getLabelOfIdx(vertex, level), -1, level};
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
        return m_nestedGraphPriorPtr->getNestedVertexCounts()[move.level].get(move.nextLabel) == 0;
    };
    bool destroyingLabel(const LabelMove<Label>& move) const {
        return move.prevLabel != move.nextLabel and m_nestedGraphPriorPtr->getNestedVertexCounts()[move.level].get(move.prevLabel) == 1 ;
    }

    int getAddedLabels(const LabelMove<Label>& move) const {
        return (int) creatingNewLabel(move) - (int) destroyingLabel(move);
    }
    bool isCreatingLabelMove(const LabelMove<Label>& move, bool reverse) const override {
        return move.addedLabels == ((reverse) ? -1 : 1);
    }
    using BaseClass = LabelProposer<Label>;
    using NestedBaseClass = NestedLabelProposer<Label>;
    using NestedBaseClass::m_nestedGraphPriorPtr;
    using NestedBaseClass::sampleLevel;
    using BaseClass::m_uniform01;
    using BaseClass::m_vertexDistribution;
    using BaseClass::m_sampleLabelCountProb;
//
public:
    using NestedLabelProposer<Label>::NestedLabelProposer;
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override {
        Level level = sampleLevel();
        Label prevLabel = m_nestedGraphPriorPtr->getLabelOfIdx(vertex, level);
        Label nextLabel = *sampleUniformlyFrom(m_emptyLabels[level].begin(), m_emptyLabels[level].end());
        LabelMove<Label> move = {vertex, prevLabel, nextLabel, 0, level};
        if ( destroyingLabel(move) )
            return {vertex, prevLabel, prevLabel, 0, level};
        move.addedLabels = 1;
        return move;
    }
    void setUpWithNestedPrior(const NestedVertexLabeledRandomGraph<Label>& graphPrior) override {
        NestedBaseClass::setUpWithNestedPrior(graphPrior);
        m_emptyLabels.clear();
        m_availableLabels.clear();
        for (Level l=0; l<m_nestedGraphPriorPtr->getDepth(); ++l){
            m_emptyLabels.push_back({m_nestedGraphPriorPtr->getNestedLabelCount(l)});
            m_availableLabels.push_back({});
            for (const auto& nr: graphPrior.getNestedVertexCounts(l))
                m_availableLabels[l].insert(nr.first);
        }
    }

    void applyLabelMove(const LabelMove<Label>& move) override {
        // graph prior must be updated before proposer
        if ( move.addedLabels==-1 and move.prevLabel != move.nextLabel )
            setUpWithNestedPrior(*m_nestedGraphPriorPtr);
        if ( move.addedLabels==1 ){
            m_availableLabels[move.level].insert(move.nextLabel);
            m_emptyLabels[move.level].erase(move.nextLabel);
            if (move.level == m_availableLabels.size() - 1){
                m_emptyLabels.push_back({1});
                m_availableLabels.push_back({0});
            }
        }
        if (m_emptyLabels[move.level].size() == 0)
            m_emptyLabels[move.level].insert(m_nestedGraphPriorPtr->getNestedLabelCount()[move.level]);
    }


};

} // namespace FastMIDyNet


#endif
