#ifndef FAST_MIDYNET_DATAMODEL_H
#define FAST_MIDYNET_DATAMODEL_H

#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"

namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class DataModel: public NestedRandomVariable{
protected:
    GraphPriorType* m_graphPriorPtr = nullptr;
    virtual void computeConsistentState() { };
    virtual void applyGraphMoveToSelf(const GraphMove& move) = 0;
public:
    DataModel(){ }
    DataModel(GraphPriorType& graphPrior){ setGraphPrior(graphPrior); }

    const MultiGraph& getGraph() const { return m_graphPriorPtr->getState(); }
    void setGraph(const MultiGraph& graph) {
        m_graphPriorPtr->setState(graph);
        computeConsistentState();
    }
    const GraphPriorType& getGraphPrior() const { return *m_graphPriorPtr; }
    GraphPriorType& getGraphPriorRef() const { return *m_graphPriorPtr; }
    void setGraphPrior(GraphPriorType& randomGraph) {
        m_graphPriorPtr = &randomGraph;
        m_graphPriorPtr->isRoot(false);
        computeConsistentState();
    }
    const size_t getSize() const { return m_graphPriorPtr->getSize(); }
    virtual void sampleState() = 0;
    void sample(){
        m_graphPriorPtr->sample();
        sampleState();
        computationFinished();
        #if DEBUG
        checkConsistency();
        #endif
    }
    void samplePrior() {
        m_graphPriorPtr->sample();
        computeConsistentState();
        computationFinished();
        #if DEBUG
        checkConsistency();
        #endif
    }
    virtual const double getLogLikelihood() const = 0;
    const double getLogPrior() const {
        return NestedRandomVariable::processRecursiveFunction<double>([&](){
            return m_graphPriorPtr->getLogJoint();
        }, 0);
    }
    const double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const = 0;
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const {
        return NestedRandomVariable::processRecursiveConstFunction<double>([&](){
            return m_graphPriorPtr->getLogJointRatioFromGraphMove(move);
        }, 0);
    }
    const double getLogJointRatioFromGraphMove(const GraphMove& move) const {
        return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
    }
    void applyGraphMove(const GraphMove& move) {
        NestedRandomVariable::processRecursiveFunction([&](){
            applyGraphMoveToSelf(move);
            m_graphPriorPtr->applyGraphMove(move);
        });
        #if DEBUG
        checkConsistency();
        #endif
    }

    void computationFinished() const override {
        m_isProcessed = false;
        m_graphPriorPtr->computationFinished();
    }
    void checkSelfSafety() const override{
        if (m_graphPriorPtr == nullptr)
            throw SafetyError("DataModel", "m_graphPriorPtr");
        m_graphPriorPtr->checkSafety();
    }

    virtual bool isSafe() const override {
        return (m_graphPriorPtr != nullptr) and (m_graphPriorPtr->isSafe());
    }

};

}
#endif
