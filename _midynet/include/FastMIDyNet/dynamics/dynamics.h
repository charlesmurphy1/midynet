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
#include "FastMIDyNet/utility/functions.h"


namespace FastMIDyNet{

class Dynamics{
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
    std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> m_vertexMapNeighborsPastStateSequence;

    void updateNeighborStateInPlace(
        BaseGraph::VertexIndex vertexIdx,
        VertexState prevVertexState,
        VertexState newVertexState,
        NeighborsState& neighborState
    ) const ;
    void updateNeighborStateMapFromEdgeMove(
        BaseGraph::Edge,
        int direction,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&
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

    const State& getCurrentState() const { return m_state; }
    const NeighborsState& getCurrentNeighborsState() const { return m_neighborsState; }
    const StateSequence& getPastStates() const { return m_pastStateSequence; }
    const StateSequence& getFutureStates() const { return m_futureStateSequence; }
    const NeighborsStateSequence& getNeighborsPastStates() const { return m_neighborsPastStateSequence; }
    const bool normalizeCoupling() const { return m_normalizeCoupling; }
    void setState(State& state) {
        m_state = state;
    }
    const MultiGraph& getGraph() const { return m_randomGraphPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) {
        m_randomGraphPtr->setGraph(graph);
        m_neighborsState = computeNeighborsState(m_state);
        if (m_pastStateSequence.size() == 0)
            return;
        for (size_t t = 0 ; t < m_numSteps ; t++){
            m_neighborsPastStateSequence[t] = computeNeighborsState(m_pastStateSequence[t]);
        }
        computeVertexNeighborsStateMap();
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
        return getCurrentState();
    }
    const State& sample(bool async=true){ return sample(getRandomState(), async); }
    void sampleState(const State& initialState, bool async=true);
    void sampleState(bool async=true){ sampleState(getRandomState(), async); }
    void sampleGraph() { setGraph(m_randomGraphPtr->sample()); }
    virtual const State getRandomState() const;
    const NeighborsState computeNeighborsState(const State& state) const;
    const VertexNeighborhoodStateSequence getVertexNeighborsState(const size_t& idx) const;
    void computeVertexNeighborsStateMap(){
        m_vertexMapNeighborsPastStateSequence;
        for (const auto& idx : getGraph()){
            m_vertexMapNeighborsPastStateSequence[idx] = getVertexNeighborsState(idx);
        }
    }

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
