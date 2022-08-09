#ifndef FAST_MIDYNET_RECONSTRUCTION_H
#define FAST_MIDYNET_RECONSTRUCTION_H

#include "FastMIDyNet/dynamics/dynamics.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/proposer/nested_label/base.hpp"
#include "FastMIDyNet/utility/maps.hpp"

namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class GraphReconstructionMCMC: public MCMC{
protected:
    Dynamics<GraphPriorType>* m_dynamicsPtr = nullptr;
    GraphPriorType* m_graphPriorPtr = nullptr;
    CallBackMap<GraphReconstructionMCMC<GraphPriorType>> m_graphCallBacks;

    double _getLogAcceptanceProbFromGraphMove(const GraphMove& move) const;
public:
    GraphReconstructionMCMC(
        Dynamics<GraphPriorType>& dynamics,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){
            m_graphCallBacks.setMCMC(*this);
            setDynamics(dynamics);
        }
    GraphReconstructionMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ m_graphCallBacks.setMCMC(*this); }

    // Accessors and mutators
    const Dynamics<GraphPriorType>& getDynamics() const { return *m_dynamicsPtr; }
    Dynamics<GraphPriorType>& getDynamicsRef() const { return *m_dynamicsPtr; }
    void setDynamics(Dynamics<GraphPriorType>& dynamics) {
        m_dynamicsPtr = &dynamics;
        m_graphPriorPtr = &m_dynamicsPtr->getGraphPriorRef();
    }
    const GraphPriorType& getGraphPrior() const { return *m_graphPriorPtr; }

    const MultiGraph& getGraph() const { return m_dynamicsPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) { m_dynamicsPtr->setGraph(graph); }

    void sample() override {
        m_dynamicsPtr->sample();
    }
    void samplePrior() override {
        m_dynamicsPtr->sampleGraph();
    }
    const double getLogLikelihood() const override { return m_dynamicsPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_dynamicsPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_dynamicsPtr->getLogJoint(); }

    // Callbacks related

    using MCMC::insertCallBack;
    void insertCallBack(std::pair<std::string, CallBack<GraphReconstructionMCMC<GraphPriorType>>*> pair) {
        m_graphCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<GraphReconstructionMCMC<GraphPriorType>>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key) override {
        if (m_mcmcCallBacks.contains(key))
            m_graphCallBacks.remove(key);
        else if( m_graphCallBacks.contains(key) )
            m_graphCallBacks.remove(key);
        else
            throw std::runtime_error("GraphReconstructionMCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<GraphReconstructionMCMC<GraphPriorType>>& getGraphCallBack(std::string key){ return m_graphCallBacks.get(key); }

    virtual void reset() override { MCMC::reset(); m_graphCallBacks.clear(); }
    virtual void onBegin() override { MCMC::onBegin(); m_graphCallBacks.onBegin(); }
    virtual void onEnd() override { MCMC::onEnd(); m_graphCallBacks.onEnd(); }
    virtual void onSweepBegin() override { MCMC::onSweepBegin(); m_graphCallBacks.onSweepBegin(); }
    virtual void onSweepEnd() override { MCMC::onSweepEnd(); m_graphCallBacks.onSweepEnd(); }
    virtual void onStepBegin() override { MCMC::onStepBegin(); m_graphCallBacks.onStepBegin(); }
    virtual void onStepEnd() override { MCMC::onStepEnd(); m_graphCallBacks.onStepEnd(); }


    // Move related
    double getLogAcceptanceProbFromGraphMove(const GraphMove& move) const {
        return processRecursiveConstFunction<double>([&](){
            return _getLogAcceptanceProbFromGraphMove(move);
        }, 0);
    }
    virtual bool doMetropolisHastingsStep() override ;

    // Debug related
    virtual bool isSafe() const override {
        return MCMC::isSafe()
        and (m_dynamicsPtr != nullptr) and (m_dynamicsPtr->isSafe());
    }

    virtual void checkSelfSafety() const override {
        MCMC::checkSelfSafety();

        if (m_dynamicsPtr == nullptr)
            throw SafetyError("GraphReconstructionMCMC, m_dynamicsPtr");
        m_dynamicsPtr->checkSafety();
    }
    virtual void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_dynamicsPtr != nullptr)
            m_dynamicsPtr->checkConsistency();
    }
    virtual void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_dynamicsPtr->computationFinished();
    }
};

using BaseBlockLabeledGraphReconstructionMCMC = GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>;

template<typename GraphPriorType>
double GraphReconstructionMCMC<GraphPriorType>::_getLogAcceptanceProbFromGraphMove(const GraphMove& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dynamicsPtr->getLogLikelihoodRatioFromGraphMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dynamicsPtr->getLogPriorRatioFromGraphMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return m_dynamicsPtr->getGraphPrior().getLogProposalRatioFromGraphMove(move) + m_lastLogJointRatio;
}

template<typename GraphPriorType>
bool GraphReconstructionMCMC<GraphPriorType>::doMetropolisHastingsStep() {
    GraphMove move = m_dynamicsPtr->getGraphPrior().proposeGraphMove();
    if (move.addedEdges == move.removedEdges)
        return m_isLastAccepted = true;
    m_lastLogAcceptance = getLogAcceptanceProbFromGraphMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_dynamicsPtr->applyGraphMove(move);
    }
    return m_isLastAccepted;
}


template<typename Label>
class VertexLabeledGraphReconstructionMCMC: public GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>{
protected:
    double m_sampleLabelProb;
    bool m_lastMoveWasLabelMove;
    using GraphPriorType = VertexLabeledRandomGraph<Label>;
    using BaseClass = GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>;
    using BaseClass::m_dynamicsPtr;
    using BaseClass::m_betaLikelihood;
    using BaseClass::m_betaPrior;
    using BaseClass::m_lastLogJointRatio;
public:

    VertexLabeledGraphReconstructionMCMC(
        Dynamics<GraphPriorType>& dynamics,
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(dynamics, betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }
    VertexLabeledGraphReconstructionMCMC(
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }

    const std::vector<Label>& getLabels() const {
        return m_dynamicsPtr->getGraphPrior().getLabels();
    }
    void setLabels(const std::vector<Label>&labels) {
        m_dynamicsPtr->getGraphPriorRef().setLabels(labels);
    }

    const double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const ;

    bool doMetropolisHastingsStep() override ;

    // Debug related

};

using BlockLabeledGraphReconstructionMCMC = VertexLabeledGraphReconstructionMCMC<BlockIndex>;

template<typename Label>
const double VertexLabeledGraphReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    return BaseClass::template processRecursiveConstFunction<double>(
        [&](){
            double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dynamicsPtr->getGraphPrior().getLogLikelihoodRatioFromLabelMove(move);
            double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dynamicsPtr->getGraphPrior().getLogPriorRatioFromLabelMove(move);
            double logProposalRatio = m_dynamicsPtr->getGraphPrior().getLogProposalRatioFromLabelMove(move);
            m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
            return logProposalRatio + m_lastLogJointRatio;
        }, 0);
}

template<typename Label>
bool VertexLabeledGraphReconstructionMCMC<Label>::doMetropolisHastingsStep() {
    m_lastMoveWasLabelMove = BaseClass::m_uniform(rng) < m_sampleLabelProb;
    if ( not m_lastMoveWasLabelMove)
        return BaseClass::doMetropolisHastingsStep();
    LabelMove<Label> move = m_dynamicsPtr->getGraphPrior().proposeLabelMove();
    if (move.prevLabel == move.nextLabel and move.addedLabels == 0)
        return BaseClass::m_isLastAccepted = true;
    BaseClass::m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    BaseClass::m_isLastAccepted = false;
    if (BaseClass::m_uniform(rng) < exp(BaseClass::m_lastLogAcceptance)){
        BaseClass::m_isLastAccepted = true;
        m_dynamicsPtr->getGraphPriorRef().applyLabelMove(move);
    }
    return BaseClass::m_isLastAccepted;
}


template<typename Label>
class NestedVertexLabeledGraphReconstructionMCMC: public GraphReconstructionMCMC<NestedVertexLabeledRandomGraph<Label>>{
protected:
    double m_sampleLabelProb;
    bool m_lastMoveWasLabelMove;

    using GraphPriorType = NestedVertexLabeledRandomGraph<Label>;
    using BaseClass = GraphReconstructionMCMC<NestedVertexLabeledRandomGraph<Label>>;
    using BaseClass::m_dynamicsPtr;
    using BaseClass::m_betaLikelihood;
    using BaseClass::m_betaPrior;
    using BaseClass::m_lastLogJointRatio;
public:

    NestedVertexLabeledGraphReconstructionMCMC(
        Dynamics<GraphPriorType>& dynamics,
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(dynamics, betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }
    NestedVertexLabeledGraphReconstructionMCMC(
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }

    const std::vector<Label>& getLabels() const {
        return m_dynamicsPtr->getGraphPrior().getLabels();
    }

    const std::vector<std::vector<Label>>& getNestedLabels() const {
        return m_dynamicsPtr->getGraphPrior().getNestedLabels();
    }
    void setNestedLabels(const std::vector<std::vector<Label>>& labels) {
        m_dynamicsPtr->getGraphPriorRef().setNestedLabels(labels);
    }

    const double getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const ;

    bool doMetropolisHastingsStep() override ;


    // Debug related
};

using NestedBlockLabeledGraphReconstructionMCMC = NestedVertexLabeledGraphReconstructionMCMC<BlockIndex>;

template<typename Label>
const double NestedVertexLabeledGraphReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove(const LabelMove<Label>& move) const {
    return BaseClass::template processRecursiveConstFunction<double>(
        [&](){
            double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dynamicsPtr->getGraphPrior().getLogLikelihoodRatioFromLabelMove(move);
            double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dynamicsPtr->getGraphPrior().getLogPriorRatioFromLabelMove(move);
            double logProposalRatio = m_dynamicsPtr->getGraphPrior().getLogProposalRatioFromLabelMove(move);
            m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
            return logProposalRatio + m_lastLogJointRatio;
        }, 0);
}

template<typename Label>
bool NestedVertexLabeledGraphReconstructionMCMC<Label>::doMetropolisHastingsStep() {
    m_lastMoveWasLabelMove = BaseClass::m_uniform(rng) < m_sampleLabelProb;
    if ( not m_lastMoveWasLabelMove)
        return BaseClass::doMetropolisHastingsStep();
    LabelMove<Label> move = m_dynamicsPtr->getGraphPrior().proposeLabelMove();
    if (move.prevLabel == move.nextLabel and move.addedLabels == 0)
        return BaseClass::m_isLastAccepted = true;
    BaseClass::m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    BaseClass::m_isLastAccepted = false;
    if (BaseClass::m_uniform(rng) < exp(BaseClass::m_lastLogAcceptance)){
        BaseClass::m_isLastAccepted = true;
        m_dynamicsPtr->getGraphPriorRef().applyLabelMove(move);
    }
    return BaseClass::m_isLastAccepted;
}



}

#endif
