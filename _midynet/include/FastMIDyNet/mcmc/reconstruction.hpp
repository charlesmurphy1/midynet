#ifndef FAST_MIDYNET_RECONSTRUCTION_H
#define FAST_MIDYNET_RECONSTRUCTION_H

#include "FastMIDyNet/dynamics/dynamics.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/utility/maps.hpp"

namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class GraphReconstructionMCMC: public MCMC{
protected:
    Dynamics<GraphPriorType>* m_dynamicsPtr = nullptr;
    EdgeProposer* m_edgeProposerPtr = nullptr;
    CallBackMap<GraphReconstructionMCMC<GraphPriorType>> m_reconCallBacks;
public:
    GraphReconstructionMCMC(
        Dynamics<GraphPriorType>& dynamics,
        EdgeProposer& edgeProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){
            setDynamics(dynamics);
            setEdgeProposer(edgeProposer);
        }
    GraphReconstructionMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ }

    // Accessors and mutators
    const Dynamics<GraphPriorType>& getDynamics() const { return *m_dynamicsPtr; }
    Dynamics<GraphPriorType>& getDynamicsRef() const { return *m_dynamicsPtr; }
    void setDynamics(Dynamics<GraphPriorType>& dynamics) {
        m_dynamicsPtr = &dynamics;
        m_dynamicsPtr->isRoot(false);
    }

    const EdgeProposer& getEdgeProposer() const { return *m_edgeProposerPtr; }
    EdgeProposer& getEdgeProposerRef() const { return *m_edgeProposerPtr; }
    void setEdgeProposer(EdgeProposer& proposer) {
        m_edgeProposerPtr = &proposer;
        m_edgeProposerPtr->isRoot(false);
    }


    const MultiGraph& getGraph() const { return m_dynamicsPtr->getGraph(); }
    const double getLogLikelihood() const override { return m_dynamicsPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_dynamicsPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_dynamicsPtr->getLogJoint(); }

    // Callbacks related
    virtual void setUp() override {
        MCMC::setUp();
        m_reconCallBacks.setUp(this);
        m_edgeProposerPtr->setUp(getGraph());
    }
    virtual void tearDown() override {
        MCMC::tearDown();
        m_reconCallBacks.tearDown();
    }

    using MCMC::insertCallBack;
    void insertCallBack(std::pair<std::string, CallBack<GraphReconstructionMCMC<GraphPriorType>>*> pair) {
        m_reconCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<GraphReconstructionMCMC<GraphPriorType>>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key, bool force=false) override {
        MCMC::removeCallBack(key, true);
        if( m_reconCallBacks.contains(key) )
            m_reconCallBacks.remove(key);
        else if ( not force)
            throw std::logic_error("GraphReconstructionMCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<GraphReconstructionMCMC<GraphPriorType>>& getCallBack(std::string key){ return m_reconCallBacks.get(key); }

    virtual void onSweepBegin() override { MCMC::onSweepBegin(); m_reconCallBacks.onSweepBegin(); }
    virtual void onSweepEnd() override { MCMC::onSweepEnd(); m_reconCallBacks.onSweepEnd(); }
    virtual void onStepBegin() override { MCMC::onStepBegin(); m_reconCallBacks.onStepBegin(); }
    virtual void onStepEnd() override { MCMC::onStepEnd(); m_reconCallBacks.onStepEnd(); }


    // Move related
    double getLogProposalProbRatioFromGraphMove(const GraphMove& move) const {
        return m_edgeProposerPtr->getLogProposalProbRatio(move);
    }
    double _getLogAcceptanceProbFromGraphMove(const GraphMove& move) const;
    double getLogAcceptanceProbFromGraphMove(const GraphMove& move) const {
        return processRecursiveFunction<double>([&](){
            return _getLogAcceptanceProbFromGraphMove(move);
        }, 0);
    }
    virtual bool doMetropolisHastingsStep() override ;

    virtual void applyGraphMove(const GraphMove& move){
        processRecursiveFunction([&](){
            m_dynamicsPtr->applyGraphMove(move);
            m_edgeProposerPtr->applyGraphMove(move);
        });
    }

    // Debug related
    virtual bool isSafe() const override {
        return MCMC::isSafe()
        and (m_dynamicsPtr != nullptr) and (m_dynamicsPtr->isSafe())
        and (m_edgeProposerPtr != nullptr)  and (m_edgeProposerPtr->isSafe());
    }

    virtual void checkSelfSafety() const override {
        if (not MCMC::isSafe())
            throw SafetyError("GraphReconstructionMCMC: it is unsafe to set up, since `MCMC` is not safe.");

        if (m_dynamicsPtr == nullptr)
            throw SafetyError("GraphReconstructionMCMC: it is unsafe to set up, since `m_dynamicsPtr` is NULL.");
        m_dynamicsPtr->checkSafety();

        if (m_edgeProposerPtr == nullptr)
            throw SafetyError("GraphReconstructionMCMC: it is unsafe to set up, since `m_edgeProposerPtr` is NULL.");
        m_edgeProposerPtr->checkSafety();
    }
    virtual void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_dynamicsPtr != nullptr)
            m_dynamicsPtr->checkConsistency();

        if (m_edgeProposerPtr != nullptr)
            m_edgeProposerPtr->checkConsistency();
    }
    virtual void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_dynamicsPtr->computationFinished();
        m_edgeProposerPtr->computationFinished();
    }
};

template<typename GraphPriorType>
double GraphReconstructionMCMC<GraphPriorType>::_getLogAcceptanceProbFromGraphMove(const GraphMove& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dynamicsPtr->getLogLikelihoodRatioFromGraphMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dynamicsPtr->getLogPriorRatioFromGraphMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return getLogProposalProbRatioFromGraphMove(move); + m_lastLogJointRatio;
}

template<typename GraphPriorType>
bool GraphReconstructionMCMC<GraphPriorType>::doMetropolisHastingsStep() {
    onStepBegin();
    GraphMove move = m_edgeProposerPtr->proposeMove();
    m_lastLogAcceptance = getLogAcceptanceProbFromGraphMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyGraphMove(move);
    }
    onStepEnd();
    return m_isLastAccepted;
}


template<typename Label>
class VertexLabeledGraphReconstructionMCMC: public GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>{
protected:
    LabelProposer<Label>* m_labelProposerPtr = nullptr;
    double m_sampleLabelProb;
    bool m_lastMoveWasLabelMove;
public:
    using GraphPriorType = VertexLabeledRandomGraph<Label>;
    using BaseClass = GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>;

    VertexLabeledGraphReconstructionMCMC(
        Dynamics<GraphPriorType>& dynamics,
        EdgeProposer& edgeProposer,
        LabelProposer<Label>& labelProposer,
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(dynamics, edgeProposer, betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){
            setLabelProposer(labelProposer);
        }
    VertexLabeledGraphReconstructionMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(betaLikelihood, betaPrior){ }

    const LabelProposer<Label>& getLabelProposer() const { return *m_labelProposerPtr; }
    LabelProposer<Label>& getLabelProposerRef() const { return *m_labelProposerPtr; }
    void setLabelProposer(LabelProposer<Label>& proposer) {
        m_labelProposerPtr = &proposer;
        m_labelProposerPtr->isRoot(false);
    }

    const std::vector<Label>& getVertexLabels() const {
        return BaseClass::m_dynamicsPtr->getGraphPrior().getVertexLabels();
    }

    void setUp() override {
        BaseClass::setUp();
        m_labelProposerPtr->setUp(BaseClass::m_dynamicsPtr->getGraphPrior());
    }


    void applyGraphMove(const GraphMove& move) {
        BaseClass::applyGraphMove(move);
        m_labelProposerPtr->applyGraphMove(move);
    }
    double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const;
    double getLogProposalProbRatioFromLabelMove(const LabelMove<Label>& move) const {
        return m_labelProposerPtr->getLogProposalProbRatio(move);
    }
    void applyLabelMove(const LabelMove<Label>& move) {
        BaseClass::m_dynamicsPtr->getGraphPriorRef().applyLabelMove(move);
        m_labelProposerPtr->applyLabelMove(move);
    }
    bool doMetropolisHastingsStep() override ;


    // Debug related
    bool isSafe() const override {
        return BaseClass::isSafe()
        and (m_labelProposerPtr != nullptr)
        and (m_labelProposerPtr->isSafe());
    }
    void checkSelfSafety() const override {
        BaseClass::checkSelfSafety();
        if (m_labelProposerPtr == nullptr)
            throw SafetyError("VertexLabeledGraphReconstructionMCMC: it is unsafe to set up, since `m_labelProposerPtr` is NULL.");
        m_labelProposerPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        BaseClass::checkSelfConsistency();
        if (m_labelProposerPtr != nullptr)
            m_labelProposerPtr->checkConsistency();
    }
    void computationFinished() const override {
        BaseClass::computationFinished();
        m_labelProposerPtr->computationFinished();
    }

};

template<typename Label>
double VertexLabeledGraphReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    double logLikelihoodRatio = (BaseClass::m_betaLikelihood == 0) ? 0 : BaseClass::m_betaLikelihood * BaseClass::m_dynamicsPtr->getGraphPrior().getLogLikelihoodRatioFromLabelMove(move);
    double logPriorRatio = (BaseClass::m_betaPrior == 0) ? 0 : BaseClass::m_betaPrior * BaseClass::m_dynamicsPtr->getGraphPrior().getLogPriorRatioFromLabelMove(move);
    BaseClass::m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
    return m_labelProposerPtr->getLogProposalProbRatio(move) + BaseClass::m_lastLogJointRatio;
}

template<typename Label>
bool VertexLabeledGraphReconstructionMCMC<Label>::doMetropolisHastingsStep() {
    BaseClass::onStepBegin();
    m_lastMoveWasLabelMove = BaseClass::m_uniform(rng) < m_sampleLabelProb;
    if ( not m_lastMoveWasLabelMove)
        BaseClass::doMetropolisHastingsStep();
    LabelMove<Label> move = m_labelProposerPtr->proposeMove();
    BaseClass::m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    BaseClass::m_isLastAccepted = false;
    if (BaseClass::m_uniform(rng) < exp(BaseClass::m_lastLogAcceptance)){
        BaseClass::m_isLastAccepted = true;
        applyLabelMove(move);
    }
    BaseClass::onStepEnd();
    return BaseClass::m_isLastAccepted;
}


}

#endif
