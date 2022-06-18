#ifndef FAST_MIDYNET_DYNAMICS_MCMC_H
#define FAST_MIDYNET_DYNAMICS_MCMC_H

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/graph_mcmc.hpp"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/utility/maps.hpp"

namespace FastMIDyNet{

class DynamicsOnRandomGraphMCMC: virtual public RandomGraphMCMC{
private:
    Dynamics* m_dynamicsPtr = nullptr;
    bool m_lastMoveWasGraphMove;
    EdgeProposer* m_edgeProposerPtr = nullptr;
    CallBackMap<DynamicsOnRandomGraphMCMC> m_dynamicsMCMCCallbacks;
public:
    DynamicsOnRandomGraphMCMC(
        Dynamics& dynamics,
        EdgeProposer& edgeProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    RandomGraphMCMC(dynamics.getRandomGraphRef(), betaLikelihood, betaPrior){
            setDynamics(dynamics);
            setEdgeProposer(edgeProposer);
        }
    DynamicsOnRandomGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    RandomGraphMCMC(betaLikelihood, betaPrior){ }

    // Accessors and mutators
    const Dynamics& getDynamics() const { return *m_dynamicsPtr; }
    Dynamics& getDynamicsRef() const { return *m_dynamicsPtr; }
    void setDynamics(Dynamics& dynamics) {
        m_dynamicsPtr = &dynamics;
        m_dynamicsPtr->isRoot(false);
        setRandomGraph(m_dynamicsPtr->getRandomGraphRef());
    }

    const EdgeProposer& getEdgeProposer() const { return *m_edgeProposerPtr; }
    EdgeProposer& getEdgeProposerRef() const { return *m_edgeProposerPtr; }
    void setEdgeProposer(EdgeProposer& proposer) { m_edgeProposerPtr = &proposer; m_edgeProposerPtr->isRoot(false); }
    GraphMove proposeEdgeMove() const { return m_edgeProposerPtr->proposeMove(); }


    const double getLogLikelihood() const override { return m_dynamicsPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_dynamicsPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_dynamicsPtr->getLogJoint(); }

    // Callbacks related
    virtual void setUp() override {
        RandomGraphMCMC::setUp();
        m_dynamicsMCMCCallbacks.setUp(this);
        m_edgeProposerPtr->setUp(getGraph());
    }
    virtual void tearDown() override {
        RandomGraphMCMC::tearDown();
        m_dynamicsMCMCCallbacks.tearDown();
    }
    void insertCallBack(std::pair<std::string, CallBack<DynamicsOnRandomGraphMCMC>*> pair) {
        m_dynamicsMCMCCallbacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<DynamicsOnRandomGraphMCMC>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key, bool force=false) override {
        MCMC::removeCallBack(key, true);
        if ( m_dynamicsMCMCCallbacks.contains(key) )
            m_dynamicsMCMCCallbacks.remove(key);
        else if ( not force)
            throw std::logic_error("DynamicsOnRandomGraphMCMC: callback of key `" + key + "` cannot be removed.");
    }


    // Move related
    double getLogProposalProbRatioFromGraphMove(const GraphMove& move) const {
        return m_edgeProposerPtr->getLogProposalProbRatio(move);
    }
    double _getLogAcceptanceProbFromGraphMove(const GraphMove& move) const;
    double getLogAcceptanceProbFromGraphMove(const GraphMove& move) const{
        return processRecursiveConstFunction<double>([&](){ return _getLogAcceptanceProbFromGraphMove(move); }, 0);
    }
    virtual bool _doMetropolisHastingsStep() override ;

    virtual void applyGraphMove(const GraphMove& move){
        processRecursiveFunction([&](){
            m_dynamicsPtr->applyGraphMove(move);
            m_edgeProposerPtr->applyGraphMove(move);
        });
    }

    // Debug related
    bool isSafe() const override {
        return RandomGraphMCMC::isSafe()
        and (m_dynamicsPtr != nullptr) and (m_dynamicsPtr->isSafe())
        and (m_edgeProposerPtr != nullptr)  and (m_edgeProposerPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (not RandomGraphMCMC::isSafe())
            throw SafetyError("DynamicsOnRandomGraphMCMC: it is unsafe to set up, since `RandomGraphMCMC` is not safe.");
        if (m_dynamicsPtr == nullptr)
            throw SafetyError("DynamicsOnRandomGraphMCMC: it is unsafe to set up, since `m_dynamicsPtr` is NULL.");
        m_dynamicsPtr->checkSafety();
        if (m_edgeProposerPtr == nullptr)
            throw SafetyError("DynamicsOnRandomGraphMCMC: it is unsafe to set up, since `m_edgeProposerPtr` is NULL.");
        m_edgeProposerPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        RandomGraphMCMC::checkSelfConsistency();
        if (m_dynamicsPtr != nullptr)
            m_dynamicsPtr->checkConsistency();

        if (m_edgeProposerPtr != nullptr)
            m_edgeProposerPtr->checkConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        RandomGraphMCMC::computationFinished();
        m_dynamicsPtr->computationFinished();
        m_edgeProposerPtr->computationFinished();
    }
};

double DynamicsOnRandomGraphMCMC::_getLogAcceptanceProbFromGraphMove(const GraphMove& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dynamicsPtr->getLogLikelihoodRatioFromGraphMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dynamicsPtr->getLogPriorRatioFromGraphMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return getLogProposalProbRatioFromGraphMove(move); + m_lastLogJointRatio;
}

bool DynamicsOnRandomGraphMCMC::_doMetropolisHastingsStep() {
    m_lastMoveWasGraphMove = true;
    GraphMove move = proposeEdgeMove();
    m_lastLogAcceptance = getLogAcceptanceProbFromGraphMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyGraphMove(move);
    }
    return m_isLastAccepted;
}


template<typename Label>
class DynanicsOnVertexLabeledRandomGraphMCMC:
public VertexLabeledRandomGraphMCMC<Label>, public DynamicsOnRandomGraphMCMC{
    double m_sampleLabelProb;
    mutable bool m_lastMoveWasLabelMove;
public:
    DynanicsOnVertexLabeledRandomGraphMCMC(
        Dynamics& dynamics,
        EdgeProposer& edgeProposer,
        LabelProposer<Label>& labelProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        double sampleLabelProb=0.1):
            VertexLabeledRandomGraphMCMC<Label>( dynamics.getRandomGraphRef(), labelProposer, betaLikelihood, betaPrior),
            DynamicsOnRandomGraphMCMC(dynamics, edgeProposer, betaLikelihood, betaPrior),
            RandomGraphMCMC(dynamics.getRandomGraphRef(), betaLikelihood, betaPrior),
            m_sampleLabelProb(sampleLabelProb)
            { }



    bool _doMetropolisHastingsStep() override {
        m_lastMoveWasLabelMove = ( m_uniform(rng) < m_sampleLabelProb );
        return (m_lastMoveWasLabelMove) ?
            VertexLabeledRandomGraphMCMC<Label>::doMetropolisHastingsStep() :
            DynamicsOnRandomGraphMCMC::doMetropolisHastingsStep();
    }

    void applyGraphMove(const GraphMove& move) override {
        processRecursiveFunction([&](){
            m_dynamicsPtr->applyGraphMove(move);
            m_edgeProposerPtr->applyGraphMove(move);
            VertexLabeledRandomGraphMCMC<Label>::m_labelProposerPtr->applyGraphMove(move);
        });
    }

    bool isSafe() const override {
        return DynamicsOnRandomGraphMCMC::isSafe() and VertexLabeledRandomGraphMCMC<Label>::isSafe();
    }

    void checkSelfSafety() const override {
        DynamicsOnRandomGraphMCMC::checkSelfSafety();
        VertexLabeledRandomGraphMCMC<Label>::checkSelfSafety();
    }
    void checkSelfConsistency() const override {
        DynamicsOnRandomGraphMCMC::checkSelfConsistency();
        VertexLabeledRandomGraphMCMC<Label>::checkSelfConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        DynamicsOnRandomGraphMCMC::computationFinished();
        VertexLabeledRandomGraphMCMC<Label>::computationFinished();
    }
};

}

#endif
