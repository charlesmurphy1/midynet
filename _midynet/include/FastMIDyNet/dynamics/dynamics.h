#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>
#include <iostream>

#include "BaseGraph/types.h"

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/dynamics/types.h"


namespace FastMIDyNet{

class Dynamics{
protected:
    size_t m_numStates;
    size_t m_numSteps;
    State m_state;
    const bool m_normalizeCoupling;
    StateSequence m_pastStateSequence;
    StateSequence m_futureStateSequence;
    RandomGraph* m_randomGraphPtr;
    NeighborsStateSequence m_neighborsStateSequence;

    void updateNeighborStateInPlace(
        BaseGraph::VertexIndex vertexIdx,
        VertexState prevVertexState,
        VertexState newVertexState,
        NeighborsState& neighborState
    ) const ;
    void updateNeighborStateMapFromEdgeMove(
        BaseGraph::Edge,
        int direction,
        std::unordered_map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&,
        std::unordered_map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&
    ) const ;
public:
    explicit Dynamics(size_t numStates, size_t numSteps, bool normalizeCoupling=true):
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_normalizeCoupling(normalizeCoupling)
        { }
    explicit Dynamics(RandomGraph& randomGraph, size_t numStates, size_t numSteps, bool normalizeCoupling=true):
        m_randomGraphPtr(&randomGraph),
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_normalizeCoupling(normalizeCoupling)
        { }

    const State& getState() const { return m_state; }
    const StateSequence& getPastStates() const { return m_pastStateSequence; }
    const StateSequence& getFutureStates() const { return m_futureStateSequence; }
    const bool normalizeCoupling() const { return m_normalizeCoupling; }
    void setState(State& state) {
        m_state = state;
    }
    const MultiGraph& getGraph() const { return m_randomGraphPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) {
        m_randomGraphPtr->setGraph(graph);
        for (size_t t = 0 ; t < m_pastStateSequence.size() ; t++){
            m_neighborsStateSequence[t] = getNeighborsState(m_pastStateSequence[t]);
        }
    }

    const RandomGraph& getRandomGraph() const { return *m_randomGraphPtr; }
    RandomGraph& getRandomGraphRef() const { return *m_randomGraphPtr; }
    void setRandomGraph(RandomGraph& randomGraph) { m_randomGraphPtr = &randomGraph; }

    const int getSize() const { return m_randomGraphPtr->getSize(); }
    const int getNumStates() const { return m_numStates; }
    const int getNumSteps() const { return m_numSteps; }
    void setNumSteps(int numSteps) { m_numSteps = numSteps; }

    const State& sample(const State& initialState, bool async=true){
        m_randomGraphPtr->sample();
        sampleState(initialState, async);
        return getState();
    }
    const State& sample(bool async=true){ return sample(getRandomState(), async); }
    void sampleState(const State& initialState, bool async=true);
    void sampleState(bool async=true){ sampleState(getRandomState(), async); }
    void sampleGraph() { setGraph(m_randomGraphPtr->sample()); }
    virtual const State getRandomState() const;
    const NeighborsState getNeighborsState(const State& state) const;
    const VertexNeighborhoodStateSequence getVertexNeighborsState(const size_t& idx) const;

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
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const;
    const double getLogJointRatioFromGraphMove(const GraphMove& move) const;
    void applyGraphMove(const GraphMove& move);

    virtual void checkSafety() const ;


};

} // namespace FastMIDyNet

#endif
