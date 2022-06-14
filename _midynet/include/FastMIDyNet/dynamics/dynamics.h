#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>
#include <iostream>

#include "BaseGraph/types.h"

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/dynamics/types.h"
#include "FastMIDyNet/utility/functions.h"


namespace FastMIDyNet{

class Dynamics: public NestedRandomVariable{
protected:
    size_t m_numStates;
    size_t m_numSteps;
    State m_state;
    std::vector<State> m_neighborsState;
    const bool m_normalizeCoupling;
    StateSequence m_pastStateSequence;
    StateSequence m_futureStateSequence;
    RandomGraph* m_randomGraphPtr = nullptr;
    NeighborsStateSequence m_neighborsPastStateSequence;

    void updateNeighborsStateInPlace(
        BaseGraph::VertexIndex vertexIdx,
        VertexState prevVertexState,
        VertexState newVertexState,
        NeighborsState& neighborsState
    ) const ;
    void updateNeighborsStateFromEdgeMove(
        BaseGraph::Edge,
        int direction,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&
    ) const ;

    void checkConsistencyOfNeighborsState() const ;
    void checkConsistencyOfNeighborsPastStateSequence() const ;
public:
    explicit Dynamics(size_t numStates, size_t numSteps, bool normalizeCoupling=true):
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_normalizeCoupling(normalizeCoupling)
        { }
    explicit Dynamics(RandomGraph& randomGraph, size_t numStates, size_t numSteps, bool normalizeCoupling=true):
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_normalizeCoupling(normalizeCoupling)
        { setRandomGraph(randomGraph); }

    const State& getCurrentState() const { return m_state; }
    const NeighborsState& getCurrentNeighborsState() const { return m_neighborsState; }
    const StateSequence& getPastStates() const { return m_pastStateSequence; }
    const StateSequence& getFutureStates() const { return m_futureStateSequence; }
    const NeighborsStateSequence& getNeighborsPastStates() const { return m_neighborsPastStateSequence; }
    const bool normalizeCoupling() const { return m_normalizeCoupling; }
    void setState(State& state) {
        m_state = state;
        m_neighborsState = computeNeighborsState(m_state);
        #if DEBUG
        checkConsistency();
        #endif
    }
    const MultiGraph& getGraph() const { return m_randomGraphPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) ;

    const RandomGraph& getRandomGraph() const { return *m_randomGraphPtr; }
    RandomGraph& getRandomGraphRef() const { return *m_randomGraphPtr; }
    void setRandomGraph(RandomGraph& randomGraph) {
        m_randomGraphPtr = &randomGraph;
        m_randomGraphPtr->isRoot(false);
    }

    const int getSize() const { return m_randomGraphPtr->getSize(); }
    const int getNumStates() const { return m_numStates; }
    const int getNumSteps() const { return m_numSteps; }
    void setNumSteps(int numSteps) { m_numSteps = numSteps; }

    const State& sample(const State& initialState, bool async=true){
        m_randomGraphPtr->sample();
        sampleState(initialState, async);
        return getCurrentState();
    }
    const State& sample(bool async=true){ return sample(getRandomState(), async); }
    void sampleState(const State& initialState, bool async=true);
    void sampleState(bool async=true){ sampleState(getRandomState(), async); }
    void sampleGraph() { setGraph(m_randomGraphPtr->sample()); }
    virtual const State getRandomState() const;
    const NeighborsState computeNeighborsState(const State& state) const;
    const NeighborsStateSequence computeNeighborsStateSequence(const StateSequence& stateSequence) const;

    void syncUpdateState();
    void asyncUpdateState(int num_updates);

    const double getLogLikelihood() const;
    const double getLogPrior() const { return m_randomGraphPtr->getLogJoint(); }
    const double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
    virtual const double getTransitionProb(
        VertexState prevVertexState,
        VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const = 0;
    const std::vector<double> getTransitionProbs(
        VertexState prevVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const;

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const;
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const {
        return m_randomGraphPtr->getLogJointRatioFromGraphMove(move);
    }
    const double getLogJointRatioFromGraphMove(const GraphMove& move) const {
        return processRecursiveConstFunction<double>([&](){
            return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
        }, 0);
    }
    void _applyGraphMove(const GraphMove& move);
    void applyGraphMove(const GraphMove& move) {
        processRecursiveFunction([&](){ _applyGraphMove(move); });
    }

    void checkSelfSafety() const override;
    void checkSelfConsistency() const override;

    void computationFinished() const override {
        m_isProcessed = false;
        m_randomGraphPtr->computationFinished();
    }

    bool isSafe() const override {
        return (m_randomGraphPtr != nullptr) and (m_randomGraphPtr->isSafe())
           and (m_state.size() != 0) and (m_pastStateSequence.size() != 0)
           and (m_futureStateSequence.size() != 0) and (m_neighborsPastStateSequence.size() != 0);
    }


};

} // namespace FastMIDyNet

#endif
