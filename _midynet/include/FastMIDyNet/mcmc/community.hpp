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
    VertexLabeledRandomGraph<Label>* m_randomGraphPtr = nullptr;
    LabelProposer<Label>* m_labelProposerPtr = nullptr;
    CallBackMap<VertexLabelMCMC<Label>> m_labelCallBacks;
public:
    VertexLabelMCMC(
        VertexLabeledRandomGraph<Label>& randomGraph,
        LabelProposer<Label>& labelProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ setRandomGraph(randomGraph); setLabelProposer(labelProposer); }
    VertexLabelMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ }

    void setRandomGraph(VertexLabeledRandomGraph<Label>& randomGraph){
        m_randomGraphPtr = &randomGraph;
    }
    const VertexLabeledRandomGraph<Label>& getRandomGraph(){ return *m_randomGraphPtr; }
    VertexLabeledRandomGraph<Label>& getRandomGraphRef(){ return *m_randomGraphPtr; }

    void setLabelProposer(LabelProposer<Label>& proposer){
        m_labelProposerPtr = &proposer;
    }
    const LabelProposer<Label>& getLabelProposer(){ return *m_labelProposerPtr; }
    LabelProposer<Label>& getLabelProposerRef(){ return *m_labelProposerPtr; }

    const std::vector<Label>& getVertexLabels() const { return m_randomGraphPtr->getVertexLabels(); }
    const double getLogLikelihood() const override { return m_randomGraphPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_randomGraphPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_randomGraphPtr->getLogJoint(); }

    // Callbacks related
    void setUp() override {
        MCMC::setUp();
        m_labelCallBacks.setUp(this);
        m_labelProposerPtr->setUp(*m_randomGraphPtr);
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
    void removeCallBack(std::string key, bool force=false) override {
        MCMC::removeCallBack(key, true);
        if( m_labelCallBacks.contains(key) )
            m_labelCallBacks.remove(key);
        else if ( not force)
            throw std::logic_error("VertexLabelMCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<VertexLabelMCMC<Label>>& getCallBack(std::string key){ return m_labelCallBacks.get(key); }

    void onSweepBegin() override { MCMC::onSweepBegin(); m_labelCallBacks.onSweepBegin(); }
    void onSweepEnd() override { MCMC::onSweepEnd(); m_labelCallBacks.onSweepEnd(); }
    void onStepBegin() override { MCMC::onStepBegin(); m_labelCallBacks.onStepBegin(); }
    void onStepEnd() override { MCMC::onStepEnd(); m_labelCallBacks.onStepEnd(); }


    // Move related
    double _getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const;
    double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
        return processRecursiveFunction<double>([&](){
            return _getLogAcceptanceProbFromLabelMove(move);
        }, 0);
    }
    bool doMetropolisHastingsStep() override ;

    void applyLabelMove(const LabelMove<Label>& move){
        processRecursiveFunction([&](){
            m_randomGraphPtr->applyLabelMove(move);
            m_labelProposerPtr->applyLabelMove(move);
        });
    }

    // Debug related
    bool isSafe() const override {
        return MCMC::isSafe()
        and (m_randomGraphPtr != nullptr) and (m_randomGraphPtr->isSafe())
        and (m_labelProposerPtr != nullptr)  and (m_labelProposerPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (not MCMC::isSafe())
            throw SafetyError("VertexLabelMCMC: it is unsafe to set up, since `MCMC` is not safe.");

        if (m_randomGraphPtr == nullptr)
            throw SafetyError("VertexLabelMCMC: it is unsafe to set up, since `m_randomGraphPtr` is NULL.");
        m_randomGraphPtr->checkSafety();

        if (m_labelProposerPtr == nullptr)
            throw SafetyError("VertexLabelMCMC: it is unsafe to set up, since `m_labelProposerPtr` is NULL.");
        m_labelProposerPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_randomGraphPtr != nullptr)
            m_randomGraphPtr->checkConsistency();

        if (m_labelProposerPtr != nullptr)
            m_labelProposerPtr->checkConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_randomGraphPtr->computationFinished();
        m_labelProposerPtr->computationFinished();
    }
};



template<typename Label>
double VertexLabelMCMC<Label>::_getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_randomGraphPtr->getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_randomGraphPtr->getLogPriorRatioFromLabelMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return m_labelProposerPtr->getLogProposalProbRatio(move); + m_lastLogJointRatio;
}

template<typename Label>
bool VertexLabelMCMC<Label>::doMetropolisHastingsStep() {
    onStepBegin();
    LabelMove<Label> move = m_labelProposerPtr->proposeMove();
    m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyLabelMove(move);
    }
    onStepEnd();
    return m_isLastAccepted;
}

class BlockMCMC: public VertexLabelMCMC<BlockIndex>{ };

}

#endif
