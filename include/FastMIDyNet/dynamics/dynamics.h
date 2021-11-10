#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H

#include <random>
#include <vector>

#include "FastMIDyNet/random_variable.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class Dynamics{

    public:
        explicit Dynamics(RandomGraph& random_graph, int num_states, RNGType rng):
            m_random_graph(random_graph),
            m_rng(rng) { }

        const StateType& getState() { return m_state; }
        const StateType& getPastStates() { return m_past_states; }
        const StateType& getFutureStates() { return m_future_states; }
        void setState(StateType& state) {m_state = state; }
        const GraphType& getGraph() { return m_random_graph.getState(); }
        void setGraph(GraphType& graph) {
            m_random_graph.setState(graph);
            for (auto t = 0 ; t < m_past_states.size() ; t++){
                m_neighbors_states[t] = getNeighborsState(m_past_states[t]);
            }
        }
        const int getNumStates() { return m_num_states; }

        const StateType& sampleState(int num_steps, const StateType& initial_state);
        const StateType& sampleState(int num_steps) { return sampleState(num_steps, getRandomState()); }
        const StateType& getRandomState() const;
        const NeighborsStateType& getNeighborsStates(const StateType& state, const GraphType& graph);
        const NeighborsStateType& getNeighborsStates(const StateType& state) { return getNeighborsState(getState(), getGraph()); }
        const NeighborsStateType& getNeighborsStates() { return getNeighborsState(state, getGraph()); }
        const NeighborsStateType& getVertexNeighborsState(const size_t& idx);
        void updateNeighborStateInPlace(size_t vertex_idx, int prev_vertex_state, int new_vertex_state, NeighborsStateType& neighbor_state);
        void updateNeighborStateMapFromEdgeMove(Edge, int, map<size_t, NeighborsStateType>&, map<size_t, NeighborsStateType>&);

        void syncUpdateState();
        void asyncUpdateState(int num_updates=1);

        const double getLogLikelihood(std::vector<StateType> past_states, std::vector<StateType> future_states, GraphType graph) const;
        const double getLogLikelihood() const { return getLogLikelihood(getState(), getGraph()); }
        const double getLogPrior(GraphType graph) const;
        const double getLogPrior() const { return getLogPrior(getGraph()); }
        virtual const double getTransitionProb(int prev_vertex_state, int next_vertex_state, std::vector<int> neighborhood_state) = 0  const;
        const std::vector<double> getTransitionProbs(int prev_vertex_state, std::vector<int> neighborhood_state) const;

        const double getLogLikelihoodRatio(GraphMove move) const;
        void applyGraphMove(phMove move);
        void doMetropolisHastingsStep(double beta = 1., double sample_graph_prior = 0.);

    protected:
        int m_num_states;
        StateType m_state;
        std::vector<StateType> m_past_states;
        std::vector<StateType> m_future_states;
        std::vector<NeighborsStateType> m_neighbors_states;
        RandomGraph& m_random_graph;
        RNGType m_rng;

};

} // namespace FastMIDyNet

#endif
