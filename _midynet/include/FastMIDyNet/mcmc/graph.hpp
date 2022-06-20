#ifndef FAST_MIDYNET_GRAPH_MCMC_H
#define FAST_MIDYNET_GRAPH_MCMC_H

#include <random>

#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class GraphMCMC: public MCMC {
protected:
    RandomGraph* m_randomGraphPtr = nullptr;
    double m_betaLikelihood, m_betaPrior;
    mutable std::uniform_real_distribution<double> m_uniform;
    CallBackMap<GraphMCMC> m_randomGraphMCMCCallBacks;
public:
    GraphMCMC(
        RandomGraph& randomGraph,
        double betaLikelihood=1,
        double betaPrior=1):
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_uniform(0., 1.) {
        setRandomGraph(randomGraph);
    }

    GraphMCMC(
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
    virtual void setRandomGraph(RandomGraph& randomGraph) {
        m_randomGraphPtr = &randomGraph;
        m_randomGraphPtr->isRoot(false);
    }

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

    using MCMC::insertCallBack;
    void insertCallBack(std::pair<std::string, CallBack<GraphMCMC>*> pair) {
        pair.second->setUp(this);
        m_randomGraphMCMCCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<GraphMCMC>& callback) { insertCallBack({key, &callback}); }

    virtual void removeCallBack(std::string key, bool force=false) override {
        MCMC::removeCallBack(key, true);
        if ( m_randomGraphMCMCCallBacks.contains(key) )
            m_randomGraphMCMCCallBacks.remove(key);
        else if ( not force)
            throw std::logic_error("GraphMCMC: callback of key `" + key + "` cannot be removed.");
    }

    // Move related
    virtual bool _doMetropolisHastingsStep() override { return false; }

    // Debug related
    virtual bool isSafe() const override {
        return (m_randomGraphPtr != nullptr) and (m_randomGraphPtr->isSafe());
    }

    virtual void checkSelfSafety() const override {
        if (m_randomGraphPtr == nullptr)
            throw SafetyError("GraphMCMC: it is unsafe to set up, since `m_randomGraphPtr` is NULL.");
        m_randomGraphPtr->checkSafety();
    };

    virtual void computationFinished() const override {
        m_isProcessed = false;
        m_randomGraphPtr->computationFinished();
    }
};

template<typename Label>
class VertexLabeledGraphMCMC: virtual public GraphMCMC{
protected:
    LabelProposer<Label>* m_labelProposerPtr = nullptr;
    VertexLabeledRandomGraph<Label>* m_vertexLabeledRandomGraphPtr = nullptr;
    CallBackMap<VertexLabeledGraphMCMC<Label>> m_vertexLabeledGraphMCMCCallbacks;
public:
    VertexLabeledGraphMCMC(
        VertexLabeledRandomGraph<Label>& randomGraph,
        LabelProposer<Label>& labelProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    GraphMCMC(randomGraph, betaLikelihood, betaPrior) {
        m_vertexLabeledRandomGraphPtr = &randomGraph;
        setLabelProposer(labelProposer);
    }
    VertexLabeledGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    GraphMCMC(betaLikelihood, betaPrior) {}

    VertexLabeledGraphMCMC(VertexLabeledGraphMCMC<Label>&& other){
        m_labelProposerPtr = other.m_labelProposerPtr;
        other.m_labelProposerPtr = nullptr;
        m_labelProposerPtr = other.m_labelProposerPtr;
        other.m_labelProposerPtr = nullptr;
        m_vertexLabeledGraphMCMCCallbacks = std::move(other.m_vertexLabeledGraphMCMCCallbacks);
    }

    VertexLabeledGraphMCMC<Label>& operator=(VertexLabeledGraphMCMC<Label>&& other){
        if (&other == this)
            return *this;
        m_labelProposerPtr = other.m_labelProposerPtr;
        other.m_labelProposerPtr = nullptr;
        m_labelProposerPtr = other.m_labelProposerPtr;
        other.m_labelProposerPtr = nullptr;
        m_vertexLabeledGraphMCMCCallbacks = std::move(other.m_vertexLabeledGraphMCMCCallbacks);
        return *this;
    }

    // Mutations and Accessors
    virtual void setRandomGraph(VertexLabeledRandomGraph<Label>& randomGraph) {
        GraphMCMC::setRandomGraph(randomGraph);
        m_vertexLabeledRandomGraphPtr = &randomGraph;
    }

    const std::vector<Label>& getVertexLabels() const { return m_vertexLabeledRandomGraphPtr->getVertexLabels(); }

    const LabelProposer<Label>& getLabelProposer() const { return *m_labelProposerPtr; }
    LabelProposer<Label>& getLabelProposerRef() const { return *m_labelProposerPtr; }
    void setLabelProposer(LabelProposer<Label>& proposer) { m_labelProposerPtr = &proposer; m_labelProposerPtr->isRoot(false); }


    // Callback related
    virtual void setUp() override {
        GraphMCMC::setUp();
        m_vertexLabeledGraphMCMCCallbacks.setUp(this);
        m_labelProposerPtr->setUp(*m_vertexLabeledRandomGraphPtr);
    }
    virtual void tearDown() override {
        GraphMCMC::tearDown();
        m_vertexLabeledGraphMCMCCallbacks.tearDown();
    }
    virtual void onSweepBegin() override { GraphMCMC::onSweepBegin(); m_vertexLabeledGraphMCMCCallbacks.onSweepBegin(); }
    virtual void onSweepEnd() override { GraphMCMC::onSweepEnd(); m_vertexLabeledGraphMCMCCallbacks.onSweepEnd(); }
    virtual void onStepBegin() override { GraphMCMC::onStepBegin(); m_vertexLabeledGraphMCMCCallbacks.onStepBegin(); }
    virtual void onStepEnd() override { GraphMCMC::onStepEnd(); m_vertexLabeledGraphMCMCCallbacks.onStepEnd(); }

    using GraphMCMC::insertCallBack;
    void insertCallBack(std::pair<std::string, CallBack<VertexLabeledGraphMCMC<Label>>*> pair) {
        pair.second->setUp(this);
        m_vertexLabeledGraphMCMCCallbacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<VertexLabeledGraphMCMC<Label>>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key, bool force=false) override {
        GraphMCMC::removeCallBack(key, true);
        if ( m_vertexLabeledGraphMCMCCallbacks.contains(key) )
            m_vertexLabeledGraphMCMCCallbacks.remove(key);
        else if ( not force)
            throw std::logic_error("VertexLabeledGraphMCMC: callback of key `" + key + "` cannot be removed.");
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
        return GraphMCMC::isSafe() and (m_labelProposerPtr != nullptr)
        and (m_labelProposerPtr->isSafe());
    }
    void checkSelfSafety() const override {
        GraphMCMC::checkSelfSafety();
        if (m_labelProposerPtr == nullptr)
            throw SafetyError("GraphMCMC: it is unsafe to set up, since `m_labelProposerPtr` is NULL.");
        m_labelProposerPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        GraphMCMC::checkSelfConsistency();
        if (m_labelProposerPtr != nullptr)
            m_labelProposerPtr->checkConsistency();
    }
    void computationFinished() const override {
        GraphMCMC::computationFinished();
        m_labelProposerPtr->computationFinished();
    }
};

template<typename Label>
double VertexLabeledGraphMCMC<Label>::_getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_vertexLabeledRandomGraphPtr->getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_vertexLabeledRandomGraphPtr->getLogPriorRatioFromLabelMove(move);
    m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
    return m_labelProposerPtr->getLogProposalProbRatio(move) + m_lastLogJointRatio;
}

template<typename Label>
bool VertexLabeledGraphMCMC<Label>::_doMetropolisHastingsStep() {
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
