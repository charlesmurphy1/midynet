#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{


class Dynamics{

    public:
        explicit Dynamics(RandomGraph& random_graph, int num_states, RNG rng):
            m_random_graph(random_graph),
            m_rng(rng) { }

        const State& getState() const { return m_state; }
        const StateSequence& getPastStates() const { return m_past_state_sequence; }
        const StateSequence& getFutureStates() const { return m_future_state_sequence; }
        void setState(State& state) {m_state = state; }
        const MultiGraph& getGraph() const { return m_random_graph.getState(); }
        void setGraph(MultiGraph& graph) {
            // m_random_graph.setState(graph);
            for (auto t = 0 ; t < m_past_state_sequence.size() ; t++){
                m_neighbors_state_sequence[t] = getNeighborsStates(m_past_state_sequence[t]);
            }
        }
        const int getSize() const { return m_random_graph.getSize(); }
        const int getNumStates() const { return m_num_states; }

        void sampleState(int num_steps, const State& initial_state, bool async);
        const State getRandomState() const;
        const NeighborsState getNeighborsStates(const State& state) const;
        const VertexNeighborhoodStateSequence getVertexNeighborsState(const size_t& idx) const;

        void syncUpdateState();
        void asyncUpdateState(int num_updates);

        const double getLogLikelihood() const;
        const double getLogPrior() const { return m_random_graph.getLogJoint(); }
        const double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
        virtual const double getTransitionProb(
            VertexState prev_vertex_state,
            VertexState next_vertex_state,
            VertexNeighborhoodState neighborhood_state
        ) const = 0;
        const std::vector<double> getTransitionProbs(
            VertexState prev_vertex_state,
            VertexNeighborhoodState neighborhood_state
        ) const;

        const double getLogJointRatio(const GraphMove& move) const;
        void applyMove(const GraphMove& move);
        void doMetropolisHastingsStep(double beta = 1., double sample_graph_prior = 0.);

    protected:
        int m_num_states;
        State m_state;
        StateSequence m_past_state_sequence;
        StateSequence m_future_state_sequence;
        RandomGraph& m_random_graph;
        NeighborsStateSequence m_neighbors_state_sequence;
        RNG m_rng;

        void updateNeighborStateInPlace(
            BaseGraph::VertexIndex vertex_idx,
            VertexState prev_vertex_state,
            VertexState new_vertex_state,
            NeighborsState& neighbor_state
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
