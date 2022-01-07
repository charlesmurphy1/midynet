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
        explicit Dynamics(RandomGraph& randomGraph, size_t numStates, size_t numSteps):
            m_randomGraph(randomGraph),
            m_numStates(numStates),
            m_numSteps(numSteps)
            { }

        const State& getState() const { return m_state; }
        const StateSequence& getPastStates() const { return m_pastStateSequence; }
        const StateSequence& getFutureStates() const { return m_futureStateSequence; }
        void setState(State& state) {m_state = state; }
        const MultiGraph& getGraph() const { return m_randomGraph.getState(); }
        void setGraph(MultiGraph& graph) {
            m_randomGraph.setState(graph);
            for (size_t t = 0 ; t < m_pastStateSequence.size() ; t++){
                m_neighborsStateSequence[t] = getNeighborsState(m_pastStateSequence[t]);
            }
        }

        const RandomGraph& getRandomGraph() const { return m_randomGraph; }

        const int getSize() const { return m_randomGraph.getSize(); }
        const int getNumStates() const { return m_numStates; }

        const State& sample(const State& initialState, bool async=true){
            m_randomGraph.sample();
            sampleState(initialState, async);
            return getState();
        }
        void sample(bool async=true){ sample(getRandomState(), async); }
        void sampleState(const State& initialState, bool async=true);
        void sampleState(bool async=true){ return sampleState(getRandomState(), async); }
        virtual const State getRandomState();
        const NeighborsState getNeighborsState(const State& state) const;
        const VertexNeighborhoodStateSequence getVertexNeighborsState(const size_t& idx) const;

        void syncUpdateState();
        void asyncUpdateState(int num_updates);

        double getLogLikelihood() const;
        double getLogPrior() const { return m_randomGraph.getLogJoint(); }
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
        RandomGraph& m_randomGraph;
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
