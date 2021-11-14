#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <random>
#include <vector>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"


namespace FastMIDyNet{


class Dynamics{

    public:
        explicit Dynamics(RandomGraph& random_graph, int num_states, RNG& rng):
            m_random_graph(random_graph),
            m_rng(rng) { }

        const State& getState() { return m_state; }
        const StateSequence& getPastStates() { return m_past_states; }
        const StateSequence& getFutureStates() { return m_future_states; }
        void setState(State& state) {m_state = state; }
        const MultiGraph& getGraph() { return m_random_graph.getState(); }
        void setGraph(MultiGraph& graph) {
            m_random_graph.setState(graph);
            for (auto t = 0 ; t < m_past_states.size() ; t++){
                m_neighbors_states[t] = getNeighborsStates(m_past_states[t]);
            }
        }
        const int getNumStates() { return m_num_states; }

        const State& sampleState(int num_steps, const State& initial_state);
        const State& sampleState(int num_steps) { return sampleState(num_steps, getRandomState()); }
        const State& getRandomState() const;
        const NeighborsState& getNeighborsStates(const State& state);
        const NeighborsState& getVertexNeighborsState(const size_t& idx);
        void updateNeighborStateInPlace(size_t vertex_idx, int prev_vertex_state, int new_vertex_state, NeighborsState& neighbor_state);
        void updateNeighborStateMapFromEdgeMove(BaseGraph::Edge, int, std::map<size_t, NeighborsState>&, std::map<size_t, NeighborsState>&);

        void syncUpdateState();
        void asyncUpdateState(int num_updates=1);

        const double getLogLikelihood() const;
        const double getLogPrior() const { return m_random_graph.getLogJoint(); }
        const double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
        virtual const double getTransitionProb(int prev_vertex_state, int next_vertex_state, std::vector<int> neighborhood_state) const = 0;
        const std::vector<double> getTransitionProbs(int prev_vertex_state, std::vector<int> neighborhood_state) const;

        const double getLogJointRatio(const GraphMove& move) const;
        void applyMove(const GraphMove& move);
        void doMetropolisHastingsStep(double beta = 1., double sample_graph_prior = 0.);

    protected:
        int m_num_states;
        State m_state;
        StateSequence m_past_states;
        StateSequence m_future_states;
        RandomGraph& m_random_graph;
        NeighborsStateSequence m_neighbors_states;
        RNG m_rng;
};

} // namespace FastMIDyNet

#endif
