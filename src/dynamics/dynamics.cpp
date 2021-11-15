#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace BaseGraph;

namespace FastMIDyNet {
    void Dynamics::sampleState(int num_steps, const State& x0, bool async){
        State state = x0;
        m_past_state_sequence.clear();
        m_future_state_sequence.clear();
        m_neighbors_state_sequence.clear();

        m_past_state_sequence.resize(num_steps);
        m_future_state_sequence.resize(num_steps);
        m_neighbors_state_sequence.resize(num_steps);

        for (size_t t = 0; t < num_steps; t++) {
            m_past_state_sequence[t] = state;
            if (async) { asyncUpdateState(getSize()); }
            else { syncUpdateState(); }
            m_future_state_sequence[t] = state;
            m_neighbors_state_sequence[t] = getNeighborsStates(state);
        }
    };

    const State& Dynamics::getRandomState() const{
        size_t N = m_random_graph.getSize();
        State rnd_state(N);
        uniform_int_distribution<int> dist(0, m_num_states - 1);

        for (size_t i = 0 ; i < N ; i ++){
            rnd_state[i] = dist(m_rng);
        }

        return rnd_state;
    };

    const NeighborsState& Dynamics::getNeighborsStates(const State& state) const {
        size_t N = m_random_graph.getSize();
        NeighborsState neighbor_states(N);
        int neighbor_idx, edge_multiplicity;

        for ( auto idx: getGraph() ){
            neighbor_states[idx].resize(m_num_states);
            for ( auto neighbor: getGraph().getNeighboursOfIdx(idx) ){
                neighbor_idx = neighbor.first;
                edge_multiplicity = neighbor.second;
                neighbor_states[ idx ][ state[neighbor_idx] ] += edge_multiplicity;
            }
        }
        return neighbor_states;
    };

    const VertexNeighborhoodStateSequence& Dynamics::getVertexNeighborsState (
        const VertexIndex& vertex_idx) const {
        int num_steps = m_past_state_sequence.size();
        VertexNeighborhoodStateSequence neighbor_state(num_steps);
        for (auto t=0; t<num_steps; t++) {
            neighbor_state[t] = m_neighbors_state_sequence[t][vertex_idx];
        }
        return neighbor_state;
    };

    void Dynamics::updateNeighborStateInPlace(
        VertexIndex vertex_idx,
        VertexState prev_vertex_state,
        VertexState new_vertex_state,
        NeighborsState& neighbor_state) const {
        int neighbor_idx, edge_multiplicity;
        for ( auto neighbor: getGraph().getNeighboursOfIdx(vertex_idx) ){
            neighbor_idx = neighbor.first;
            edge_multiplicity = neighbor.second;
            neighbor_state[neighbor_idx][prev_vertex_state] -= edge_multiplicity;
            neighbor_state[neighbor_idx][new_vertex_state] += edge_multiplicity;
        }
    };

    void Dynamics::syncUpdateState(){
        State future_state(m_state);
        NeighborsState neighbor_state = getNeighborsStates(m_state);
        vector<double> trans_probs(m_num_states);

        for (auto idx: getGraph()){
            trans_probs = getTransitionProbs(m_state[idx], neighbor_state[idx]);
            future_state[idx] = generateCategorical(trans_probs, m_rng);
        }
    };

    void Dynamics::asyncUpdateState(int num_updates){
        size_t N = m_random_graph.getSize();
        VertexState new_vertex_state;
        State current_state(m_state);
        NeighborsState neighbor_state = getNeighborsStates(m_state);
        vector<double> trans_probs(m_num_states);
        uniform_int_distribution<VertexIndex> idx_generator(0, N-1);

        for (auto i=0; i < num_updates; i++){
            VertexIndex idx = idx_generator(m_rng);
            trans_probs = getTransitionProbs(current_state[idx], neighbor_state[idx]);
            new_vertex_state = generateCategorical(trans_probs, m_rng);
            updateNeighborStateInPlace(idx, current_state[idx], new_vertex_state, neighbor_state);
            current_state[idx] = new_vertex_state;
        }
    };

    const double Dynamics::getLogLikelihood() const {
        // setGraph(graph);
        double log_likelihood = 0;
        vector<int> neighbor_state(getNumStates(), 0);
        int neighbor_idx, edge_multiplicity;
        for (size_t t = 0; t < m_past_state_sequence.size(); t++){
            for (auto idx: getGraph()){
                fill(neighbor_state.begin(), neighbor_state.end(), 0);
                for (auto neighbor: getGraph().getNeighboursOfIdx(idx)){
                    neighbor_idx = neighbor.first;
                    edge_multiplicity = neighbor.second;
                    neighbor_state[m_past_state_sequence[t][idx]] += edge_multiplicity;
                }
                log_likelihood += log(getTransitionProb(m_past_state_sequence[t][idx],
                    m_future_state_sequence[t][idx], neighbor_state));
            }
        }
        return log_likelihood;
    };

    const std::vector<double> Dynamics::getTransitionProbs(VertexState prev_vertex_state, VertexNeighborhoodState neighborhood_state) const{
        std::vector<double> trans_probs(getNumStates());
        for (VertexState next_vertex_state = 0; next_vertex_state < getNumStates(); next_vertex_state++) {
            trans_probs[next_vertex_state] = getTransitionProb(prev_vertex_state, next_vertex_state, neighborhood_state);
        }
        return trans_probs;
    };

    void Dynamics::updateNeighborStateMapFromEdgeMove(Edge edge, int direction, map<VertexIndex, VertexNeighborhoodStateSequence>& prev_neighbor_map, map<VertexIndex, VertexNeighborhoodStateSequence>& next_neighbor_map) const{
        int num_steps = m_past_state_sequence.size();
        VertexIndex v_idx = edge.first, u_idx = edge.second;
        if (prev_neighbor_map.count(v_idx) == 0){
            prev_neighbor_map[v_idx] = getVertexNeighborsState(v_idx) ;
            next_neighbor_map[v_idx] = getVertexNeighborsState(v_idx) ;
        }
        else if (prev_neighbor_map.count(u_idx) == 0){
            prev_neighbor_map[u_idx] = getVertexNeighborsState(u_idx);
            next_neighbor_map[u_idx] = getVertexNeighborsState(u_idx) ;
        }

        VertexState v_state, u_state;
        for (size_t t = 0; t < num_steps; t++) {
            u_state = m_past_state_sequence[t][u_idx];
            v_state = m_past_state_sequence[t][v_idx];
            next_neighbor_map[u_idx][t][v_state] += direction;
            next_neighbor_map[v_idx][t][u_state] += direction;
        }
    };

    const double Dynamics::getLogJointRatio(const GraphMove& move) const{
        int num_steps = m_past_state_sequence.size();
        double log_likelihood_ratio = 0;
        set<size_t> vertices_affected;
        map<VertexIndex,VertexNeighborhoodStateSequence> prev_neighbor_map, next_neighbor_map;

        size_t v_idx, u_idx;
        for (const auto& edge : move.edges_added){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_map, next_neighbor_map);
        }
        for (const auto& edge : move.edges_removed){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_map, next_neighbor_map);
        }

        for (const auto& idx: vertices_affected){
            for (size_t t = 0; t < num_steps; t++) {
                log_likelihood_ratio += log(getTransitionProb(m_past_state_sequence[t][idx], m_future_state_sequence[t][idx], next_neighbor_map[idx][t]));
                log_likelihood_ratio -= log(getTransitionProb(m_past_state_sequence[t][idx], m_future_state_sequence[t][idx], prev_neighbor_map[idx][t]));
            }
        }

        return log_likelihood_ratio;
    }

    void Dynamics::applyMove(const GraphMove& move){
        int num_steps = m_past_state_sequence.size();
        set<VertexIndex> vertices_affected;
        map<VertexIndex, VertexNeighborhoodStateSequence> prev_neighbor_map, next_neighbor_map;

        VertexNeighborhoodStateSequence neighbor_state(num_steps);
        size_t v_idx, u_idx;
        for (const auto& edge : move.edges_added){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_map, next_neighbor_map);
        }
        for (const auto& edge : move.edges_removed){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_map, next_neighbor_map);
        }

        for (const auto& idx: vertices_affected){
            for (size_t t = 0; t < num_steps; t++) {
                m_neighbors_state_sequence[t][idx] = next_neighbor_map[idx][t];
            }
        }
    }


}
