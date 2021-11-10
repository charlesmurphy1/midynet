#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/dynamics/generators.h"
#include "FastMIDyNet/dynamics/types.h"

using namespace std;

namespace FastMIDyNet {
    const StateType& Dynamics::sampleState(int num_steps, StateType x0, bool async=true){
        state = x0;
        m_past_states.clear();
        m_future_states.clear();
        m_neighbors_states.clear();

        m_past_states.resize(num_steps);
        m_future_states.resize(num_steps);
        m_neighbors_states.resize(num_steps);

        for (size_t t = 0; t < num_steps; t++) {
            past_states[t] = state;
            if (async) { asyncUpdateState(); }
            else { syncUpdateState(); }
            m_future_states[t] = state;
            m_neighbors_states[t] = getNeighborsStates(state);
        }
    };

    const StateType& Dynamics::getRandomState() const{
        int N = random_graph.getNumberOfVertices();
        StateType rnd_state(N);
        uniform_int_distribution<int> dist(0, num_states - 1);

        for (size_t i = 0 ; i < N ; i ++){
            rnd_state[i] = dist(m_rng);
        }

        return rnd_state;
    }

    const NeighborsStateType& Dynamics::getNeighborsStates(StateType state, GraphType graph){
        int N = graph.getSize();
        NeighborsStateType neighbor_states(N, 0);
        int neighbor_idx, edge_multiplicity;

        for (auto idx: graph){
            neighbor_states[idx].resize(num_states);
            for (auto auto& neighbour: graph.getNeighboursOfIdx(idx)){
                neighbor_idx = neighbors.first;
                edge_multiplicity = neighbors.second;
                neighbor_states[state[neighbor_idx]] += edge_multiplicity;
            }
        }
        return neighbor_states;
    }

    const NeighborsStateType& Dynamics::getVertexNeighborsState(size_t idx){
        int num_steps = m_past_states.size();
        NeighborsStateType neighbor_state(num_steps, 0);
        for (auto t=0; t<num_steps; t++) {
            neighbor_state[t] = m_neighbors_states[t][idx];
        }
        return neighbor_state;
    }

    void Dynamics::updateNeighborStateInPlace(size_t vertex_idx, int prev_state, int new_state, NeighborsStateType& neighbor_state){
        int neighbor_idx, edge_multiplicity;
        for (auto auto& neighbour: getGraph().getNeighboursOfIdx(vertex_idx)){
            neighbor_idx = neighbors.first;
            edge_multiplicity = neighbors.second;
            neighbor_state[neighbor_idx][prev_state] -= edge_multiplicity;
            neighbor_state[neighbor_idx][new_state] += edge_multiplicity;
        }
    }


    void Dynamics::syncUpdateState(){
        StateType future_state(state);
        NeighborsStateType neighbor_state = getNeighborsStates(state);
        vector<double> trans_probs(num_states);

        for (auto idx: getGraph()){
            trans_probs = getTransitionProb(state[idx], neighbor_state[idx]);
            future_state[idx] = generateCategorical(trans_probs, m_rng);
        }
    }

    void Dynamics::asyncUpdateState(int num_updates=1){
        int N = random_graph.getNumberOfVertices();
        int new_vertex_state;
        StateType current_state(state);
        NeighborsStateType neighbor_state = getNeighborsStates(state);
        vector<double> trans_probs(num_states);
        uniform_int_distribution<size_t> idx_generator(0, N-1);

        for (auto i=0; i < num_updates; i++){
            idx = idx_generator(m_rng);
            trans_probs = getTransitionProb(current_state[idx], neighbor_state[idx]);
            new_vertex_state = generateCategorical(trans_probs, m_rng);
            updateNeighborStateInPlace(idx, current_state[idx], new_vertex_state, neighbor_state);
            current_state[idx] = new_vertex_state;
        }
    }

    const double Dynamics::getLogLikelihood(vector<StateType> past_states, vector<StateType> future_states, GraphType graph){
        // setGraph(graph);
        double log_likelihood = 0;
        vector<int> neighbor_state(getNumStates(), 0);
        int neighbor_idx, edge_multiplicity;
        for (size_t t = 0; t < past_states.size(); t++){
            for (auto& idx: graph){
                fill(neighbor_state.begin(), neighbor_state.end(), 0);
                for (auto&neighbor: graph.getNeighboursOfIdx(idx)){
                    neighbor_idx = neighbor.first;
                    edge_multiplicity = neighbor.second;
                    neighbor_state[past_states[t][idx]] += edge_multiplicity;
                }
                log_likelihood += log(getTransitionProb(past_states[t][idx], future_states[t][idx], neighbor_state));
            }
        }
        return log_likelihood;
    }

    const std::vector<double> Dynamics::getTransitionProbs(int prev_vertex_state, std::vector<int> neighborhood_state){
        std::vector<double> trans_probs(getNumStates());
        for (size_t next_state = 0; next_state < getNumStates(); next_state++) {
            trans_probs[next_state] = getTransitionProb(prev_vertex_state, next_vertex_state, neighborhood_state);
        }
        return trans_probs;
    };

    void Dynamics::updateNeighborStateMapFromEdgeMove(Edge edge, int move_type, map<size_t, NeighborsStateType>& prev_neighbor_state_map, map<size_t, NeighborsStateType>& neighbor_state_map){
        int num_steps = m_past_states.size();
        size_t v_idx = edge.first, u_idx = edge.second;
        if (vertices_affected.count(v_idx) == 0){
            prev_neighbor_state_map.insert(v_idx, getVertexNeighborsState(v_idx));
            next_neighbor_state_map.insert(v_idx, getVertexNeighborsState(v_idx));
        }
        else if (vertices_affected.count(u_idx) == 0){
            prev_neighbor_state_map.insert(u_idx, getVertexNeighborsState(u_idx));
            next_neighbor_state_map.insert(u_idx, getVertexNeighborsState(u_idx));
        }

        int v_state, u_state;
        for (size_t t = 0; t < num_steps; t++) {
            u_state = m_past_states[t][u_idx];
            v_state = m_past_states[t][v_idx];
            next_neighbor_state_map[u_idx][t][v_state] += direction;
            next_neighbor_state_map[v_idx][t][u_state] += direction;
        }
    };

    const double Dynamics::getLogLikelihoodRatio(GraphMove move){
        int num_steps = m_past_states.size();
        double log_likelihood_ratio = 0;
        set<size_t> vertices_affected;
        map<size_t,NeighborsStateType> prev_neighbor_state_map, next_neighbor_state_map;

        NeighborsStateType neighbor_state(num_steps);
        size_t v_idx, u_idx;
        for (const auto& edge : move.edges_added){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_state_map, next_neighbor_state_map);
        }
        for (const auto& edges : move.edges_removed){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_state_map, next_neighbor_state_map);
        }

        for (const auto& idx: vertices_affected){
            for (size_t t = 0; t < num_steps; t++) {
                log_likelihood_ratio += log(getTransitionProb(m_past_states[t][idx], m_future_states[t][idx], next_neighbor_state_map[idx][t]));
                log_likelihood_ratio -= log(getTransitionProb(m_past_states[t][idx], m_future_states[t][idx], prev_neighbor_state_map[idx][t]));
            }
        }

        return log_likelihood_ratio;
    }

    void Dynamics::applyGraphMove(GraphMove move){
        int num_steps = m_past_states.size();
        set<size_t> vertices_affected;
        map<size_t,NeighborsStateType> prev_neighbor_state_map, next_neighbor_state_map;

        NeighborsStateType neighbor_state(num_steps);
        size_t v_idx, u_idx;
        for (const auto& edge : move.edges_added){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_state_map, next_neighbor_state_map);
        }
        for (const auto& edges : move.edges_removed){
            v_idx = edge.first;
            u_idx = edge.second;
            vertices_affected.insert(v_idx);
            vertices_affected.insert(u_idx);
            updateNeighborStateMapFromEdgeMove(edge, 1, prev_neighbor_state_map, next_neighbor_state_map);
        }

        for (const auto& idx: vertices_affected){
            for (size_t t = 0; t < num_steps; t++) {
                m_neighbors_states[t][idx] = next_neighbor_state_map[idx][t];
            }
        }
    }


}
