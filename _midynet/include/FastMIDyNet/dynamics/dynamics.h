#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>
#include <iostream>

#include "BaseGraph/types.h"

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/dynamics/types.h"


namespace FastMIDyNet{

class Dynamics{

    public:
        explicit Dynamics(size_t numStates, size_t numSteps):
            m_numStates(numStates),
            m_numSteps(numSteps)
            { }
        explicit Dynamics(RandomGraph& randomGraph, size_t numStates, size_t numSteps):
            m_randomGraphPtr(&randomGraph),
            m_numStates(numStates),
            m_numSteps(numSteps)
            { }

        const State& getState() const { return m_state; }
        const StateSequence& getPastStates() const { return m_pastStateSequence; }
        const StateSequence& getFutureStates() const { return m_futureStateSequence; }
        void setState(State& state) {m_state = state; }
        const MultiGraph& getGraph() const { return m_randomGraphPtr->getState(); }
        void setGraph(const MultiGraph& graph) {
            m_randomGraphPtr->setState(graph);
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
        virtual const State getRandomState();
        const NeighborsState getNeighborsState(const State& state) const;
        const VertexNeighborhoodStateSequence getVertexNeighborsState(const size_t& idx) const;

        void syncUpdateState();
        void asyncUpdateState(int num_updates);

        double getLogLikelihood() const;
        double getLogPrior() const { return m_randomGraphPtr->getLogJoint(); }
        double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
        virtual double getTransitionProb(
            VertexState prevVertexState,
            VertexState nextVertexState,
            VertexNeighborhoodState neighborhoodState
        ) const = 0;
        const std::vector<double> getTransitionProbs(
            VertexState prevVertexState,
            VertexNeighborhoodState neighborhoodState
        ) const;

        double getLogLikelihoodRatio(const GraphMove& move) const;
        double getLogPriorRatio(const GraphMove& move);
        double getLogJointRatio(const GraphMove& move);
        void applyMove(const GraphMove& move);

    protected:
        size_t m_numStates;
        size_t m_numSteps;
        State m_state;
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
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&
        ) const ;

};

} // namespace FastMIDyNet

#endif
