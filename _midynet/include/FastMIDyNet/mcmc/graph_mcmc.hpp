#ifndef FAST_MIDYNET_GRAPH_MCMC_H
#define FAST_MIDYNET_GRAPH_MCMC_H

#include <random>

#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/label_proposer/label_proposer.hpp"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class RandomGraphMCMC: public MCMC {
protected:
    RandomGraph* m_randomGraphPtr = nullptr;
    double m_betaLikelihood, m_betaPrior;
    mutable std::uniform_real_distribution<double> m_uniform;
    CallBackMap<RandomGraphMCMC> m_randomGraphMCMCCallBacks;
public:
    RandomGraphMCMC(
        RandomGraph& randomGraph,
        double betaLikelihood=1,
        double betaPrior=1):
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_uniform(0., 1.) {
        setRandomGraph(randomGraph);
    }

    RandomGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    m_uniform(0., 1.) {}

    // Accessors
    const MultiGraph& getGraph() const { return m_randomGraphPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) { m_randomGraphPtr->setGraph(graph); setUp(); }
    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }
    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    const RandomGraph& getRandomGraph() const { return *m_randomGraphPtr; }
    RandomGraph& getRandomGraphRef() const { return *m_randomGraphPtr; }
    virtual void setRandomGraph(RandomGraph& randomGraph) { m_randomGraphPtr = &randomGraph; m_randomGraphPtr->isRoot(false);}

    const double getLogLikelihood() const override { return m_randomGraphPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_randomGraphPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_randomGraphPtr->getLogJoint(); }

    // Callback related
    virtual void setUp() override {
        MCMC::setUp();
        m_randomGraphMCMCCallBacks.setUp(this);
    }
    virtual void tearDown() override {
        MCMC::tearDown();
        m_randomGraphMCMCCallBacks.tearDown();
    }
    virtual void onSweepBegin() override { MCMC::onSweepBegin(); m_randomGraphMCMCCallBacks.onSweepBegin(); }
    virtual void onSweepEnd() override { MCMC::onSweepEnd(); m_randomGraphMCMCCallBacks.onSweepEnd(); }
    virtual void onStepBegin() override { MCMC::onStepBegin(); m_randomGraphMCMCCallBacks.onStepBegin(); }
    virtual void onStepEnd() override { MCMC::onStepEnd(); m_randomGraphMCMCCallBacks.onStepEnd(); }
    void insertCallBack(std::pair<std::string, CallBack<RandomGraphMCMC>*> pair) {
        pair.second->setUp(this);
        m_randomGraphMCMCCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<RandomGraphMCMC>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key, bool force=false) override {
        MCMC::removeCallBack(key, true);
        if ( m_randomGraphMCMCCallBacks.contains(key) )
            m_randomGraphMCMCCallBacks.remove(key);
        else if ( not force)
            throw std::logic_error("RandomGraphMCMC: callback of key `" + key + "` cannot be removed.");
    }

    // Move related
    virtual bool _doMetropolisHastingsStep() override { return false; }

    // Debug related
    virtual bool isSafe() const override {
        return (m_randomGraphPtr != nullptr) and (m_randomGraphPtr->isSafe());
    }

    virtual void checkSelfSafety() const override {
        if (m_randomGraphPtr == nullptr)
            throw SafetyError("RandomGraphMCMC: it is unsafe to set up, since `m_randomGraphPtr` is NULL.");
        m_randomGraphPtr->checkSafety();
    };

    virtual void computationFinished() const override {
        m_isProcessed = false;
        m_randomGraphPtr->computationFinished();
    }
};

template<typename Label>
class VertexLabeledRandomGraphMCMC: virtual public RandomGraphMCMC{
    LabelProposer<Label>* m_labelProposerPtr = nullptr;
    VertexLabeledRandomGraph<Label>* m_vertexLabeledRandomGraphPtr = nullptr;
    CallBackMap<VertexLabeledRandomGraphMCMC<Label>> m_vertexLabeledRandomGraphMCMCCallbacks;
public:
    VertexLabeledRandomGraphMCMC(
        VertexLabeledRandomGraph<Label>& randomGraph,
        EdgeProposer& edgeProposer,
        LabelProposer<Label>& labelProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    RandomGraphMCMC(randomGraph, edgeProposer, betaLikelihood, betaPrior) {
        m_vertexLabeledRandomGraphPtr = &randomGraph;
        setLabelProposer(labelProposer);
    }
    VertexLabeledRandomGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    RandomGraphMCMC(betaLikelihood, betaPrior) {}

    // Mutations and Accessors
    void setRandomGraph(VertexLabeledRandomGraph<Label>& randomGraph) {
        RandomGraphMCMC::setRandomGraph(randomGraph); m_vertexLabeledRandomGraphPtr = &randomGraph;
    }

    const std::vector<Label>& getVertexLabels() const { return m_vertexLabeledRandomGraphPtr->getVertexLabels(); }

    const LabelProposer<Label>& getLabelProposer() const { return *m_labelProposerPtr; }
    LabelProposer<Label>& getLabelProposerRef() const { return *m_labelProposerPtr; }
    void setLabelProposer(LabelProposer<Label>& proposer) { m_labelProposerPtr = &proposer; m_labelProposerPtr->isRoot(false); }


    // Callback related
    virtual void setUp() override {
        RandomGraphMCMC::setUp();
        m_vertexLabeledRandomGraphMCMCCallbacks.setUp(this);
        m_labelProposerPtr->setUp(*m_vertexLabeledRandomGraphPtr);
    }
    virtual void tearDown() override {
        RandomGraphMCMC::tearDown();
        m_vertexLabeledRandomGraphMCMCCallbacks.tearDown();
    }
    virtual void onSweepBegin() override { RandomGraphMCMC::onSweepBegin(); m_vertexLabeledRandomGraphMCMCCallbacks.onSweepBegin(); }
    virtual void onSweepEnd() override { RandomGraphMCMC::onSweepEnd(); m_vertexLabeledRandomGraphMCMCCallbacks.onSweepEnd(); }
    virtual void onStepBegin() override { RandomGraphMCMC::onStepBegin(); m_vertexLabeledRandomGraphMCMCCallbacks.onStepBegin(); }
    virtual void onStepEnd() override { RandomGraphMCMC::onStepEnd(); m_vertexLabeledRandomGraphMCMCCallbacks.onStepEnd(); }

    void insertCallBack(std::pair<std::string, CallBack<VertexLabeledRandomGraphMCMC<Label>>*> pair) {
        pair.second->setUp(this);
        m_vertexLabeledRandomGraphMCMCCallbacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<VertexLabeledRandomGraphMCMC<Label>>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key, bool force=false) override {
        RandomGraphMCMC::removeCallBack(key, true);
        if ( m_vertexLabeledRandomGraphMCMCCallbacks.contains(key) )
            m_vertexLabeledRandomGraphMCMCCallbacks.remove(key);
        else if ( not force)
            throw std::logic_error("VertexLabeledRandomGraphMCMC: callback of key `" + key + "` cannot be removed.");
    }

    // Move related
    double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
        return processRecursiveConstFunction<double>([&](){ return _getLogAcceptanceProbFromLabelMove(move); }, 0) ;
    }
    double _getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const;
    double getLogProposalProbRatioFromLabelMove(const LabelMove<Label>& move) const { return m_labelProposerPtr->getLogProposalProbRatio(move); }
    void applyLabelMove(const LabelMove<Label>& move) {
        processRecursiveFunction([&](){
            m_labelProposerPtr->applyLabelMove(move);
            m_vertexLabeledRandomGraphPtr->applyLabelMove(move);
        });
    }
    virtual bool _doMetropolisHastingsStep() override ;


    // Debug related
    bool isSafe() const override {
        return RandomGraphMCMC::isSafe() and (m_labelProposerPtr != nullptr)
        and (m_labelProposerPtr->isSafe());
    }
    void checkSelfSafety() const override {
        RandomGraphMCMC::checkSelfSafety();
        if (m_labelProposerPtr == nullptr)
            throw SafetyError("RandomGraphMCMC: it is unsafe to set up, since `m_labelProposerPtr` is NULL.");
        m_labelProposerPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        RandomGraphMCMC::checkSelfConsistency();
        if (m_labelProposerPtr != nullptr)
            m_labelProposerPtr->checkConsistency();
    }
    void computationFinished() const override {
        RandomGraphMCMC::computationFinished();
        m_labelProposerPtr->computationFinished();
    }
};

template<typename Label>
double VertexLabeledRandomGraphMCMC<Label>::_getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_vertexLabeledRandomGraphPtr->getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_vertexLabeledRandomGraphPtr->getLogPriorRatioFromLabelMove(move);
    m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
    return m_labelProposerPtr->getLogProposalProbRatio(move) + m_lastLogJointRatio;
}

template<typename Label>
bool VertexLabeledRandomGraphMCMC<Label>::_doMetropolisHastingsStep() {
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
