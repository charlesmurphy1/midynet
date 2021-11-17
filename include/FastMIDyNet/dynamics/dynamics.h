#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/dynamics/types.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{

class Dynamics{

    public:
        explicit Dynamics(RandomGraph& randomGraph, int numStates):
            m_randomGraph(randomGraph),
            m_numStates(numStates)
            { }

        const State& getState() const { return m_state; }
        const StateSequence& getPastStates() const { return m_pastStateSequence; }
        const StateSequence& getFutureStates() const { return m_futureStateSequence; }
        void setState(State& state) {m_state = state; }
        const MultiGraph& getGraph() const { return m_randomGraph.getState(); }
        void setGraph(MultiGraph& graph) {
            m_randomGraph.setState(graph);
            for (auto t = 0 ; t < m_pastStateSequence.size() ; t++){
                m_neighborsStateSequence[t] = getNeighborsState(m_pastStateSequence[t]);
            }
        }
        const int getSize() const { return m_randomGraph.getSize(); }
        const int getNumStates() const { return m_numStates; }

        void sampleState(int numSteps, const State& initialState, bool async=true);
        void sampleState(int numSteps, bool async=true){ return sampleState(numSteps, getRandomState(), async); }
        const State getRandomState();
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

        double getLogJointRatio(const GraphMove& move) const;
        void applyMove(const GraphMove& move);
        void doMetropolisHastingsStep(double beta = 1., double sampleGraphPrior = 0.);

    protected:
        int m_numStates;
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
