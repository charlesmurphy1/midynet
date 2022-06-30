#ifndef FAST_MIDYNET_COMMUNITY_H
#define FAST_MIDYNET_COMMUNITY_H

#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "mcmc.h"

namespace FastMIDyNet{

template<typename Label>
class VertexLabelMCMC: public MCMC{
protected:
    VertexLabeledRandomGraph<Label>* m_graphPriorPtr = nullptr;
    LabelProposer<Label>* m_labelProposerPtr = nullptr;
    CallBackMap<VertexLabelMCMC<Label>> m_labelCallBacks;

    double _getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const;
public:
    VertexLabelMCMC(
        VertexLabeledRandomGraph<Label>& graphPrior,
        LabelProposer<Label>& labelProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ setGraphPrior(graphPrior); setLabelProposer(labelProposer); }
    VertexLabelMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ }

    void setGraphPrior(VertexLabeledRandomGraph<Label>& graphPrior){
        m_graphPriorPtr = &graphPrior;
    }
    const VertexLabeledRandomGraph<Label>& getGraphPrior(){ return *m_graphPriorPtr; }
    VertexLabeledRandomGraph<Label>& getGraphPriorRef(){ return *m_graphPriorPtr; }

    void setLabelProposer(LabelProposer<Label>& proposer){
        m_labelProposerPtr = &proposer;
    }
    const LabelProposer<Label>& getLabelProposer(){ return *m_labelProposerPtr; }
    LabelProposer<Label>& getLabelProposerRef(){ return *m_labelProposerPtr; }


    const MultiGraph& getGraph() const { return m_graphPriorPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) {m_graphPriorPtr->setGraph(graph); }

    const std::vector<Label>& getLabels() const { return m_graphPriorPtr->getLabels(); }
    void setLabels(const std::vector<Label>& labels) { m_graphPriorPtr->setLabels(labels); }


    void sample() override { m_graphPriorPtr->sample(); }
    void samplePrior() override { m_graphPriorPtr->sampleLabels(); }
    const double getLogLikelihood() const override { return m_graphPriorPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_graphPriorPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_graphPriorPtr->getLogJoint(); }

    // Callbacks related
    void setUp() override {
        MCMC::setUp();
        m_labelCallBacks.setUp(this);
        m_labelProposerPtr->setUp(*m_graphPriorPtr);
    }
    void tearDown() override {
        MCMC::tearDown();
        m_labelCallBacks.tearDown();
    }

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
            throw std::logic_error("VertexLabelMCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<VertexLabelMCMC<Label>>& getLabelCallBack(std::string key){ return m_labelCallBacks.get(key); }

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

    void applyLabelMove(const LabelMove<Label>& move){
        processRecursiveFunction([&](){
            m_graphPriorPtr->applyLabelMove(move);
            m_labelProposerPtr->applyLabelMove(move);
        });
    }

    // Debug related
    bool isSafe() const override {
        return MCMC::isSafe()
        and (m_graphPriorPtr != nullptr) and (m_graphPriorPtr->isSafe())
        and (m_labelProposerPtr != nullptr)  and (m_labelProposerPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (not MCMC::isSafe())
            throw SafetyError("VertexLabelMCMC: it is unsafe to set up, since `MCMC` is not safe.");

        if (m_graphPriorPtr == nullptr)
            throw SafetyError("VertexLabelMCMC: it is unsafe to set up, since `m_graphPriorPtr` is NULL.");
        m_graphPriorPtr->checkSafety();

        if (m_labelProposerPtr == nullptr)
            throw SafetyError("VertexLabelMCMC: it is unsafe to set up, since `m_labelProposerPtr` is NULL.");
        m_labelProposerPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_graphPriorPtr != nullptr)
            m_graphPriorPtr->checkConsistency();

        if (m_labelProposerPtr != nullptr)
            m_labelProposerPtr->checkConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_graphPriorPtr->computationFinished();
        m_labelProposerPtr->computationFinished();
    }
};

using BlockLabelMCMC =  VertexLabelMCMC<BlockIndex>;


template<typename Label>
double VertexLabelMCMC<Label>::_getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_graphPriorPtr->getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_graphPriorPtr->getLogPriorRatioFromLabelMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return m_labelProposerPtr->getLogProposalProbRatio(move) + m_lastLogJointRatio;
}

template<typename Label>
bool VertexLabelMCMC<Label>::doMetropolisHastingsStep() {
    LabelMove<Label> move = m_labelProposerPtr->proposeMove();
    m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyLabelMove(move);
    }
    return m_isLastAccepted;
}


}

#endif
