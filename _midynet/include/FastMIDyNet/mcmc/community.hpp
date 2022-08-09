#ifndef FAST_MIDYNET_COMMUNITY_H
#define FAST_MIDYNET_COMMUNITY_H

#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/proposer/nested_label/base.hpp"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "mcmc.h"

namespace FastMIDyNet{

template<typename Label>
class VertexLabelMCMC: public MCMC{
protected:
    VertexLabeledRandomGraph<Label>* m_graphPriorPtr = nullptr;
    CallBackMap<VertexLabelMCMC<Label>> m_labelCallBacks;

    double _getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const;
public:
    VertexLabelMCMC(
        VertexLabeledRandomGraph<Label>& graphPrior,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ m_labelCallBacks.setMCMC(*this); setGraphPrior(graphPrior); }
    VertexLabelMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ m_labelCallBacks.setMCMC(*this); }

    void setGraphPrior(VertexLabeledRandomGraph<Label>& graphPrior){
        m_graphPriorPtr = &graphPrior;
    }
    const VertexLabeledRandomGraph<Label>& getGraphPrior() const { return *m_graphPriorPtr; }
    VertexLabeledRandomGraph<Label>& getGraphPriorRef(){ return *m_graphPriorPtr; }

    const MultiGraph& getGraph() const { return m_graphPriorPtr->getState(); }
    void setGraph(const MultiGraph& graph) {m_graphPriorPtr->setState(graph); }

    const std::vector<Label>& getLabels() const { return m_graphPriorPtr->getLabels(); }
    void setLabels(const std::vector<Label>& labels) { m_graphPriorPtr->setLabels(labels); }


    void sample() override { m_graphPriorPtr->sample(); }
    void samplePrior() override { m_graphPriorPtr->sampleLabels(); }
    const double getLogLikelihood() const override { return m_graphPriorPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_graphPriorPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_graphPriorPtr->getLogJoint(); }

    // Callbacks related
    using MCMC::insertCallBack;
    void insertCallBack(std::pair<std::string, CallBack<VertexLabelMCMC<Label>>*> pair) {
        m_labelCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<VertexLabelMCMC<Label>>& callback) { insertCallBack({key, &callback}); }
    void removeCallBack(std::string key) override {
        if ( m_mcmcCallBacks.contains(key) )
            m_mcmcCallBacks.remove(key);
        if( m_labelCallBacks.contains(key) )
            m_labelCallBacks.remove(key);
        else
            throw std::runtime_error("VertexLabelMCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<VertexLabelMCMC<Label>>& getLabelCallBack(std::string key){ return m_labelCallBacks.get(key); }

    virtual void reset() { MCMC::reset(); m_labelCallBacks.clear(); }
    void onBegin() override { MCMC::onBegin(); m_labelCallBacks.onBegin(); }
    void onEnd() override { MCMC::onEnd(); m_labelCallBacks.onEnd(); }
    void onSweepBegin() override { MCMC::onSweepBegin(); m_labelCallBacks.onSweepBegin(); }
    void onSweepEnd() override { MCMC::onSweepEnd(); m_labelCallBacks.onSweepEnd(); }
    void onStepBegin() override { MCMC::onStepBegin(); m_labelCallBacks.onStepBegin(); }
    void onStepEnd() override { MCMC::onStepEnd(); m_labelCallBacks.onStepEnd(); }


    // Move related
    double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
        return processRecursiveFunction<double>([&](){
            return _getLogAcceptanceProbFromLabelMove(move);
        }, 0);
    }
    bool doMetropolisHastingsStep() override ;

    // Debug related
    bool isSafe() const override {
        return MCMC::isSafe()
        and (m_graphPriorPtr != nullptr) and (m_graphPriorPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (m_graphPriorPtr == nullptr)
            throw SafetyError("VertexLabelMCMC", "m_graphPriorPtr");
        m_graphPriorPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_graphPriorPtr != nullptr)
            m_graphPriorPtr->checkConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_graphPriorPtr->computationFinished();
    }
};

using BlockLabelMCMC =  VertexLabelMCMC<BlockIndex>;


template<typename Label>
double VertexLabelMCMC<Label>::_getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_graphPriorPtr->getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_graphPriorPtr->getLogPriorRatioFromLabelMove(move);
    double logProposalRatio = m_graphPriorPtr->getLogProposalRatioFromLabelMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return logProposalRatio + m_lastLogJointRatio;
}

template<typename Label>
bool VertexLabelMCMC<Label>::doMetropolisHastingsStep() {
    LabelMove<Label> move = m_graphPriorPtr->proposeLabelMove();
    if (move.prevLabel == move.nextLabel and move.addedLabels == 0)
        return m_isLastAccepted = true;
    m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_graphPriorPtr->applyLabelMove(move);
    }
    return m_isLastAccepted;
}

template<typename Label>
class NestedVertexLabelMCMC: public MCMC{
protected:
    NestedVertexLabeledRandomGraph<Label>* m_graphPriorPtr = nullptr;
    CallBackMap<NestedVertexLabelMCMC<Label>> m_labelCallBacks;

    double _getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const;
public:
    NestedVertexLabelMCMC(
        NestedVertexLabeledRandomGraph<Label>& graphPrior,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ m_labelCallBacks.setMCMC(*this); setGraphPrior(graphPrior); }
    NestedVertexLabelMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ m_labelCallBacks.setMCMC(*this); }

    void setGraphPrior(NestedVertexLabeledRandomGraph<Label>& graphPrior){
        m_graphPriorPtr = &graphPrior;
    }
    const NestedVertexLabeledRandomGraph<Label>& getGraphPrior() const { return *m_graphPriorPtr; }
    NestedVertexLabeledRandomGraph<Label>& getGraphPriorRef(){ return *m_graphPriorPtr; }

    const MultiGraph& getGraph() const { return m_graphPriorPtr->getState(); }
    void setGraph(const MultiGraph& graph) {m_graphPriorPtr->setState(graph); }

    const std::vector<Label>& getLabels() const { return m_graphPriorPtr->getLabels(); }
    const std::vector<std::vector<Label>>& getNestedLabels() const { return m_graphPriorPtr->getNestedLabels(); }
    void setNestedLabels(const std::vector<std::vector<Label>>& labels) { m_graphPriorPtr->setNestedLabels(labels); }


    void sample() override { m_graphPriorPtr->sample(); }
    void samplePrior() override { m_graphPriorPtr->sampleLabels(); }
    const double getLogLikelihood() const override { return m_graphPriorPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_graphPriorPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_graphPriorPtr->getLogJoint(); }

    // Callbacks related

    using MCMC::insertCallBack;
    void insertCallBack(std::pair<std::string, CallBack<NestedVertexLabelMCMC<Label>>*> pair) {
        m_labelCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<NestedVertexLabelMCMC<Label>>& callback) { insertCallBack({key, &callback}); }
    void removeCallBack(std::string key) override {
        if ( m_mcmcCallBacks.contains(key) )
            m_mcmcCallBacks.remove(key);
        if( m_labelCallBacks.contains(key) )
            m_labelCallBacks.remove(key);
        else
            throw std::runtime_error("NestedVertexLabelMCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<NestedVertexLabelMCMC<Label>>& getLabelCallBack(std::string key){ return m_labelCallBacks.get(key); }


    void reset() { MCMC::reset(); m_labelCallBacks.clear(); }
    void onBegin() override { MCMC::onBegin(); m_labelCallBacks.onBegin(); }
    void onEnd() override { MCMC::onEnd(); m_labelCallBacks.onEnd(); }
    void onSweepBegin() override { MCMC::onSweepBegin(); m_labelCallBacks.onSweepBegin(); }
    void onSweepEnd() override { MCMC::onSweepEnd(); m_labelCallBacks.onSweepEnd(); }
    void onStepBegin() override { MCMC::onStepBegin(); m_labelCallBacks.onStepBegin(); }
    void onStepEnd() override { MCMC::onStepEnd(); m_labelCallBacks.onStepEnd(); }


    // Move related
    double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
        return processRecursiveFunction<double>([&](){
            return _getLogAcceptanceProbFromLabelMove(move);
        }, 0);
    }

    bool doMetropolisHastingsStep() override ;

    // Debug related
    bool isSafe() const override {
        return MCMC::isSafe()
        and (m_graphPriorPtr != nullptr) and (m_graphPriorPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (m_graphPriorPtr == nullptr)
            throw SafetyError("NestedVertexLabelMCMC", "m_graphPriorPtr");
        m_graphPriorPtr->checkSafety();

    }
    void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_graphPriorPtr != nullptr)
            m_graphPriorPtr->checkConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_graphPriorPtr->computationFinished();
    }
};

using NestedBlockLabelMCMC =  NestedVertexLabelMCMC<BlockIndex>;


template<typename Label>
double NestedVertexLabelMCMC<Label>::_getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_graphPriorPtr->getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_graphPriorPtr->getLogPriorRatioFromLabelMove(move);
    double logProposalRatio = m_graphPriorPtr->getLogProposalRatioFromLabelMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return logProposalRatio + m_lastLogJointRatio;
}

template<typename Label>
bool NestedVertexLabelMCMC<Label>::doMetropolisHastingsStep() {
    LabelMove<Label> move = m_graphPriorPtr->proposeLabelMove();
    if (not m_graphPriorPtr->isValidLabelMove(move))
        return m_isLastAccepted = false;
    if (move.prevLabel == move.nextLabel and move.addedLabels == 0)
        return m_isLastAccepted = true;
    m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_graphPriorPtr->applyLabelMove(move);
    }
    return m_isLastAccepted;
}




}

#endif
