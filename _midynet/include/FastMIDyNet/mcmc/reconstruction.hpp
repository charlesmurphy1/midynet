#ifndef FAST_MIDYNET_RECONSTRUCTION_H
#define FAST_MIDYNET_RECONSTRUCTION_H

#include "FastMIDyNet/data/data_model.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/proposer/nested_label/base.hpp"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/data/types.h"

namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class GraphReconstructionMCMC: public MCMC{
protected:
    DataModel<GraphPriorType>* m_dataModelPtr = nullptr;
    GraphPriorType* m_graphPriorPtr = nullptr;
    CallBackMap<GraphReconstructionMCMC<GraphPriorType>> m_graphCallBacks;

    double _getLogAcceptanceProbFromGraphMove(const GraphMove& move) const;
public:
    GraphReconstructionMCMC(
        DataModel<GraphPriorType>& dataModel,
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){
            m_graphCallBacks.setMCMC(*this);
            setDataModel(dataModel);
        }
    GraphReconstructionMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    MCMC(betaLikelihood, betaPrior){ m_graphCallBacks.setMCMC(*this); }

    // Accessors and mutators
    const DataModel<GraphPriorType>& getDataModel() const { return *m_dataModelPtr; }
    DataModel<GraphPriorType>& getDataModelRef() const { return *m_dataModelPtr; }
    void setDataModel(DataModel<GraphPriorType>& dataModel) {
        m_dataModelPtr = &dataModel;
        m_graphPriorPtr = &m_dataModelPtr->getGraphPriorRef();
    }
    const GraphPriorType& getGraphPrior() const { return *m_graphPriorPtr; }
    GraphPriorType& getGraphPriorRef() const { return *m_graphPriorPtr; }

    const MultiGraph& getGraph() const { return m_dataModelPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) { m_dataModelPtr->setGraph(graph); }

    void sample() override {
        m_dataModelPtr->sample();
    }
    void sampleState() override {
        m_dataModelPtr->sampleState();
    }
    void samplePrior() override {
        m_dataModelPtr->samplePrior();
    }
    const double getLogLikelihood() const override { return m_dataModelPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_dataModelPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_dataModelPtr->getLogJoint(); }

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
        and (m_dataModelPtr != nullptr) and (m_dataModelPtr->isSafe());
    }

    virtual void checkSelfSafety() const override {
        MCMC::checkSelfSafety();

        if (m_dataModelPtr == nullptr)
            throw SafetyError("GraphReconstructionMCMC", "m_dataModelPtr");
        m_dataModelPtr->checkSafety();
    }
    virtual void checkSelfConsistency() const override {
        MCMC::checkSelfConsistency();
        if (m_dataModelPtr != nullptr)
            m_dataModelPtr->checkConsistency();
    }
    virtual void computationFinished() const override {
        m_isProcessed = false;
        MCMC::computationFinished();
        m_dataModelPtr->computationFinished();
    }
};

template<typename GraphPriorType>
double GraphReconstructionMCMC<GraphPriorType>::_getLogAcceptanceProbFromGraphMove(const GraphMove& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dataModelPtr->getLogLikelihoodRatioFromGraphMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dataModelPtr->getLogPriorRatioFromGraphMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return m_dataModelPtr->getGraphPrior().getLogProposalRatioFromGraphMove(move) + m_lastLogJointRatio;
}

template<typename GraphPriorType>
bool GraphReconstructionMCMC<GraphPriorType>::doMetropolisHastingsStep() {
    GraphMove move = m_dataModelPtr->getGraphPrior().proposeGraphMove();
    if (not m_graphPriorPtr->isValidGraphMove(move))
        return m_isLastAccepted = false;
    if (move.addedEdges == move.removedEdges)
        return m_isLastAccepted = true;
    m_lastLogAcceptance = getLogAcceptanceProbFromGraphMove(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_dataModelPtr->applyGraphMove(move);
    }
    return m_isLastAccepted;
}

using GraphReconstructionMCMCBase = GraphReconstructionMCMC<>;
using BlockLabeledGraphReconstructionMCMCBase = GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>;
using NestedBlockLabeledGraphReconstructionMCMCBase = GraphReconstructionMCMC<NestedVertexLabeledRandomGraph<BlockIndex>>;


template<typename Label>
class VertexLabeledGraphReconstructionMCMC: public GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>{
protected:
    double m_sampleLabelProb;
    bool m_lastMoveWasLabelMove;
    using GraphPriorType = VertexLabeledRandomGraph<Label>;
    using BaseClass = GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>;
    using BaseClass::m_dataModelPtr;
    using BaseClass::m_betaLikelihood;
    using BaseClass::m_betaPrior;
    using BaseClass::m_lastLogJointRatio;
public:

    VertexLabeledGraphReconstructionMCMC(
        DataModel<GraphPriorType>& dataModel,
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(dataModel, betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }
    VertexLabeledGraphReconstructionMCMC(
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }
    void onSweepEnd() override {
        // BaseClass::m_graphPriorPtr->reduceLabels();
        MCMC::onSweepEnd();
        BaseClass::m_graphCallBacks.onSweepEnd();
    }

    const std::vector<Label>& getLabels() const {
        return m_dataModelPtr->getGraphPrior().getLabels();
    }
    void setLabels(const std::vector<Label>&labels, bool reduce=false) {
        m_dataModelPtr->getGraphPriorRef().setLabels(labels, reduce);
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
            double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dataModelPtr->getGraphPrior().getLogLikelihoodRatioFromLabelMove(move);
            double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dataModelPtr->getGraphPrior().getLogPriorRatioFromLabelMove(move);
            double logProposalRatio = m_dataModelPtr->getGraphPrior().getLogProposalRatioFromLabelMove(move);
            m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
            return logProposalRatio + m_lastLogJointRatio;
        }, 0);
}

template<typename Label>
bool VertexLabeledGraphReconstructionMCMC<Label>::doMetropolisHastingsStep() {
    m_lastMoveWasLabelMove = BaseClass::m_uniform(rng) < m_sampleLabelProb;
    if ( not m_lastMoveWasLabelMove)
        return BaseClass::doMetropolisHastingsStep();
    LabelMove<Label> move = m_dataModelPtr->getGraphPrior().proposeLabelMove();
    if (not m_dataModelPtr->getGraphPrior().isValidLabelMove(move))
        return BaseClass::m_isLastAccepted = false;
    if (move.prevLabel == move.nextLabel and move.addedLabels == 0)
        return BaseClass::m_isLastAccepted = true;
    BaseClass::m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    BaseClass::m_isLastAccepted = false;
    if (BaseClass::m_uniform(rng) < exp(BaseClass::m_lastLogAcceptance)){
        BaseClass::m_isLastAccepted = true;
        m_dataModelPtr->getGraphPriorRef().applyLabelMove(move);
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
    using BaseClass::m_dataModelPtr;
    using BaseClass::m_betaLikelihood;
    using BaseClass::m_betaPrior;
    using BaseClass::m_lastLogJointRatio;
public:

    NestedVertexLabeledGraphReconstructionMCMC(
        DataModel<GraphPriorType>& dataModel,
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(dataModel, betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }
    NestedVertexLabeledGraphReconstructionMCMC(
        double sampleLabelProb=0.5,
        double betaLikelihood=1,
        double betaPrior=1):
    BaseClass(betaLikelihood, betaPrior),
    m_sampleLabelProb(sampleLabelProb){ }


    void onSweepEnd() override {
        // BaseClass::m_graphPriorPtr->reduceLabels();
        MCMC::onSweepEnd();
        BaseClass::m_graphCallBacks.onSweepEnd();
    }
    const std::vector<Label>& getLabels() const {
        return m_dataModelPtr->getGraphPrior().getLabels();
    }

    const std::vector<std::vector<Label>>& getNestedLabels() const {
        return m_dataModelPtr->getGraphPrior().getNestedLabels();
    }
    void setNestedLabels(const std::vector<std::vector<Label>>& labels, bool reduce=false) {
        m_dataModelPtr->getGraphPriorRef().setNestedLabels(labels, reduce);
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
            double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dataModelPtr->getGraphPrior().getLogLikelihoodRatioFromLabelMove(move);
            double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dataModelPtr->getGraphPrior().getLogPriorRatioFromLabelMove(move);
            double logProposalRatio = m_dataModelPtr->getGraphPrior().getLogProposalRatioFromLabelMove(move);
            m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
            return logProposalRatio + m_lastLogJointRatio;
        }, 0);
}

template<typename Label>
bool NestedVertexLabeledGraphReconstructionMCMC<Label>::doMetropolisHastingsStep() {
    m_lastMoveWasLabelMove = BaseClass::m_uniform(rng) < m_sampleLabelProb;
    if ( not m_lastMoveWasLabelMove)
        return BaseClass::doMetropolisHastingsStep();
    LabelMove<Label> move = m_dataModelPtr->getGraphPrior().proposeLabelMove();
    if (move.prevLabel == move.nextLabel and move.addedLabels == 0)
        return BaseClass::m_isLastAccepted = true;
    BaseClass::m_lastLogAcceptance = getLogAcceptanceProbFromLabelMove(move);
    BaseClass::m_isLastAccepted = false;
    if (BaseClass::m_uniform(rng) < exp(BaseClass::m_lastLogAcceptance)){
        BaseClass::m_isLastAccepted = true;
        m_dataModelPtr->getGraphPriorRef().applyLabelMove(move);
    }
    return BaseClass::m_isLastAccepted;
}



}

#endif
