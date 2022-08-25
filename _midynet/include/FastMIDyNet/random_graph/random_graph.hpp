#ifndef FAST_MIDYNET_RANDOM_GRAPH_H
#define FAST_MIDYNET_RANDOM_GRAPH_H

// #include <random>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"

#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/proposer/nested_label/base.hpp"

namespace FastMIDyNet{

// class EdgeProposer;
//
// template<typename Label>
// class LabelProposer;
//
// template<typename Label>
// class NestedLabelProposer;

class RandomGraph: public NestedRandomVariable{
private:
    int m_samplingIteration=0;
    int m_maxIteration=100;
protected:
    GraphLikelihoodModel* m_likelihoodModelPtr = nullptr;
    EdgeProposer* m_edgeProposerPtr = nullptr;
    bool m_withSelfLoops, m_withParallelEdges;
    size_t m_size;
    MultiGraph m_state;
    virtual void _applyGraphMove(const GraphMove&);
    virtual const double _getLogPrior() const { return 0; }
    virtual const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const { return 0; }
    virtual void sampleOnlyPrior() { };
    virtual void setUpLikelihood() {
        m_likelihoodModelPtr->m_statePtr = &m_state;
    }
    virtual void setUp();
    virtual void computeConsistentState() { }

public:
    RandomGraph(size_t size, bool withSelfLoops=true, bool withParallelEdges=true):
        m_size(size), m_state(size),
        m_withSelfLoops(withSelfLoops),
        m_withParallelEdges(withParallelEdges) { }

    RandomGraph(
        size_t size,
        GraphLikelihoodModel& likelihoodModel,
        bool withSelfLoops=true,
        bool withParallelEdges=true):
            m_size(size), m_state(size),
            m_likelihoodModelPtr(&likelihoodModel),
            m_withSelfLoops(withSelfLoops),
            m_withParallelEdges(withParallelEdges) { }

    const MultiGraph& getState() const { return m_state; }

    virtual void setState(const MultiGraph& state) {
        m_state = state;
        computeConsistentState();
        setUp();
    }
    const size_t getSize() const { return m_size; }
    void setSize(const size_t size) { m_size = size; }
    virtual const size_t getEdgeCount() const = 0;
    const double getAverageDegree() const {
        double avgDegree = 2 * (double) getEdgeCount();
        avgDegree /= (double) getSize();
        return avgDegree;
    }
    const bool withSelfLoops() const { return m_withSelfLoops; }
    const bool withSelfLoops(bool condition) { return m_withSelfLoops = condition; }
    const bool withParallelEdges() const { return m_withParallelEdges; }
    const bool withParallelEdges(bool condition) { return m_withParallelEdges = condition; }

    void setEdgeProposer(EdgeProposer& proposer) {
        proposer.isRoot(false);
        m_edgeProposerPtr = &proposer;
    }
    const EdgeProposer& getEdgeProposer() {
        return *m_edgeProposerPtr;
    }
    EdgeProposer& getEdgeProposerRef() {
        return *m_edgeProposerPtr;
    }

    void sample() {
        try{
            processRecursiveFunction([&](){ sampleOnlyPrior(); });
            sampleState();
            setUp();
        } catch (std::invalid_argument){
            if (m_samplingIteration < m_maxIteration){
                ++m_samplingIteration;
                sample();
            }else
                throw std::runtime_error("RandomGraph: could not sample the model after " + std::to_string(m_maxIteration) + " iterations.");
        }
        m_samplingIteration=0;
    }
    void sampleState() {
        setState(m_likelihoodModelPtr->sample());
        computationFinished();
    }
    void samplePrior() {
        processRecursiveFunction([&](){
            sampleOnlyPrior();
            computeConsistentState();
        });
    }

    const double getLogLikelihood() const { return m_likelihoodModelPtr->getLogLikelihood(); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const { return m_likelihoodModelPtr->getLogLikelihoodRatioFromGraphMove(move); }
    const double getLogProposalRatioFromGraphMove (const GraphMove& move) const;

    const double getLogPrior() const {
        return processRecursiveFunction<double>([&](){ return _getLogPrior();}, 0);
    }
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const {
        return processRecursiveConstFunction<double>([&](){return _getLogPriorRatioFromGraphMove(move);}, 0);
    }

    const double getLogJoint() const {
        return getLogLikelihood() + getLogPrior();
    }
    const double getLogJointRatioFromGraphMove (const GraphMove& move) const{
        return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
    }

    void applyGraphMove(const GraphMove& move) ;
    const GraphMove proposeGraphMove() const ;

    virtual const bool isCompatible(const MultiGraph& graph) const { return graph.getSize() == m_size; }
    virtual bool isSafe() const override { return m_likelihoodModelPtr and m_likelihoodModelPtr->isSafe(); }
    virtual void checkSelfSafety() const override {
        if (m_likelihoodModelPtr == nullptr)
            throw SafetyError("RandomGraph", "m_likelihoodModelPtr");
        if (m_edgeProposerPtr == nullptr)
            throw SafetyError("RandomGraph", "m_edgeProposerPtr");
    }
    virtual void checkSelfConsistency() const override {
        m_likelihoodModelPtr->checkConsistency();
    }

    virtual bool isValidGraphMove(const GraphMove& move) const { return true; }
};

template <typename Label>
class VertexLabeledRandomGraph: public RandomGraph{
protected:
    LabelProposer<Label>* m_labelProposerPtr = nullptr;
    virtual void _applyLabelMove(const LabelMove<Label>&) { };
    virtual const double _getLogPriorRatioFromLabelMove (const LabelMove<Label>& move) const { return 0; }
    VertexLabeledGraphLikelihoodModel<Label>* m_vertexLabeledlikelihoodModelPtr = nullptr;

    using RandomGraph::m_edgeProposerPtr;
    virtual void setUp() override ;

public:
    VertexLabeledRandomGraph(size_t size, bool withSelfLoops=true, bool withParallelEdges=true):
        RandomGraph(size, withSelfLoops, withParallelEdges){ }
    VertexLabeledRandomGraph(
        size_t size, VertexLabeledGraphLikelihoodModel<Label>& likelihoodModel,
        bool withSelfLoops=true, bool withParallelEdges=true):
            RandomGraph(size, likelihoodModel, withSelfLoops, withParallelEdges),
            m_vertexLabeledlikelihoodModelPtr(&likelihoodModel){ }
    virtual const std::vector<Label>& getLabels() const = 0;
    virtual const size_t getLabelCount() const = 0;
    virtual const CounterMap<Label>& getVertexCounts() const = 0;
    virtual const CounterMap<Label>& getEdgeLabelCounts() const = 0;
    virtual const LabelGraph& getLabelGraph() const = 0;
    const Label& getLabelOfIdx(BaseGraph::VertexIndex vertexIdx) const { return getLabels()[vertexIdx]; }

    virtual void setLabels(const std::vector<Label>&, bool reduce=false) = 0;
    virtual void sampleOnlyLabels() = 0;



    void setLabelProposer(LabelProposer<Label>& proposer) {
        proposer.isRoot(false);
        m_labelProposerPtr = &proposer;
    }
    const LabelProposer<Label>& getLabelProposer() const {
        return *m_labelProposerPtr;
    }
    LabelProposer<Label>& getLabelProposerRef() {
        return *m_labelProposerPtr;
    }

    virtual const double getLabelLogJoint() const = 0;

    const double getLogLikelihoodRatioFromLabelMove (const LabelMove<Label>& move) const {
        return m_vertexLabeledlikelihoodModelPtr->getLogLikelihoodRatioFromLabelMove(move);
    }
    const double getLogPriorRatioFromLabelMove (const LabelMove<Label>& move) const {
        return processRecursiveConstFunction<double>([&]() { return _getLogPriorRatioFromLabelMove(move); }, 0);
    }
    const double getLogJointRatioFromLabelMove (const LabelMove<Label>& move) const{
        return getLogPriorRatioFromLabelMove(move) + getLogLikelihoodRatioFromLabelMove(move);
    }
    const double getLogProposalRatioFromLabelMove (const LabelMove<Label>& move) const;

    void applyLabelMove(const LabelMove<Label>& move) ;
    const LabelMove<Label> proposeLabelMove() const;
    virtual bool isValidLabelMove(const LabelMove<Label>& move) const { return true; }
    virtual void checkSelfSafety() const override {
        RandomGraph::checkSelfSafety();
        if (m_labelProposerPtr == nullptr)
            throw SafetyError("RandomGraph", "m_labelProposerPtr");
    }
    virtual void checkSelfConsistency() const override {
        m_edgeProposerPtr->checkConsistency();
    }
    virtual void reduceLabels() { }
};

using BlockLabeledRandomGraph = VertexLabeledRandomGraph<BlockIndex>;


template <typename Label>
class NestedVertexLabeledRandomGraph: public VertexLabeledRandomGraph<Label>{
protected:
    NestedLabelProposer<Label>* m_nestedLabelProposerPtr = nullptr;
    using VertexLabeledRandomGraph<Label>::m_state;
    using VertexLabeledRandomGraph<Label>::m_edgeProposerPtr;
    using VertexLabeledRandomGraph<Label>::m_labelProposerPtr;
    virtual void setUp() override ;
public:
    using VertexLabeledRandomGraph<Label>::VertexLabeledRandomGraph;

    void setLabels(const std::vector<Label>&, bool reduce=false) override {
        throw DepletedMethodError("NestedVertexLabeledRandomGraph", "setLabels");
    }
    virtual void setNestedLabels(const std::vector<std::vector<Label>>&, bool reduce=false) = 0;

    virtual const size_t getDepth() const = 0;

    virtual const Label getLabelOfIdx(BaseGraph::VertexIndex vertexIdx, Level level) const = 0;
    virtual const Label getNestedLabelOfIdx(BaseGraph::VertexIndex vertexIdx, Level level) const = 0;
    virtual const std::vector<std::vector<Label>>& getNestedLabels() const = 0;
    virtual const std::vector<Label>& getNestedLabels(Level) const = 0;
    virtual const std::vector<size_t>& getNestedLabelCount() const = 0;
    virtual const size_t getNestedLabelCount(Level) const = 0;
    virtual const std::vector<CounterMap<Label>>& getNestedVertexCounts() const = 0;
    virtual const CounterMap<Label>& getNestedVertexCounts(Level) const = 0;
    virtual const std::vector<CounterMap<Label>>& getNestedEdgeLabelCounts() const = 0;
    virtual const CounterMap<Label>& getNestedEdgeLabelCounts(Level) const = 0;
    virtual const std::vector<MultiGraph>& getNestedLabelGraph() const = 0;
    virtual const MultiGraph& getNestedLabelGraph(Level ) const = 0;

    void setNestedLabelProposer(NestedLabelProposer<Label>& proposer) {
        proposer.isRoot(false);
        m_labelProposerPtr = &proposer;
        m_nestedLabelProposerPtr = &proposer;
    }
    const NestedLabelProposer<Label>& getNestedLabelProposer() {
        return *m_nestedLabelProposerPtr;
    }
    NestedLabelProposer<Label>& getNestedLabelProposerRef() {
        return *m_nestedLabelProposerPtr;
    }

    using VertexLabeledRandomGraph<Label>::getLabelOfIdx;
    const std::vector<Label>& getLabels() const override { return getNestedLabels()[0]; }
    const size_t getLabelCount() const override { return getNestedLabelCount()[0]; }
    const CounterMap<Label>& getVertexCounts() const override { return getNestedVertexCounts()[0]; }
    const CounterMap<Label>& getEdgeLabelCounts() const override { return getNestedEdgeLabelCounts()[0]; }
    const MultiGraph& getLabelGraph() const override { return getNestedLabelGraph()[0]; }
    virtual void checkSelfConsistency() const override {
        RandomGraph::checkSelfConsistency();
        m_labelProposerPtr->checkConsistency();
    }
};
using NestedBlockLabeledRandomGraph = NestedVertexLabeledRandomGraph<BlockIndex>;


template<typename Label>
void VertexLabeledRandomGraph<Label>::setUp() {
    m_edgeProposerPtr->setUpWithPrior(*this);
    m_labelProposerPtr->setUpWithPrior(*this);
}

template<typename Label>
const double VertexLabeledRandomGraph<Label>::getLogProposalRatioFromLabelMove (const LabelMove<Label>& move) const{
    return m_labelProposerPtr->getLogProposalProbRatio(move);
}

template<typename Label>
void VertexLabeledRandomGraph<Label>::applyLabelMove(const LabelMove<Label>& move) {
    processRecursiveFunction([&](){ _applyLabelMove(move); });
    m_labelProposerPtr->applyLabelMove(move);
    #if DEBUG
    checkConsistency();
    #endif
}

template<typename Label>
const LabelMove<Label> VertexLabeledRandomGraph<Label>::proposeLabelMove() const {
    return m_labelProposerPtr->proposeMove();
}


template<typename Label>
void NestedVertexLabeledRandomGraph<Label>::setUp() {
    m_edgeProposerPtr->setUpWithPrior(*this);
    m_nestedLabelProposerPtr->setUpWithNestedPrior(*this);
}

} // namespace FastMIDyNet

#endif
