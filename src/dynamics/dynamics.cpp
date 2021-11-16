#include <chrono>
#include <cmath>
#include <map>

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace BaseGraph;

namespace FastMIDyNet {
void Dynamics::sampleState(int numSteps, const State& x0, bool async){
    State state = x0;
    m_pastStateSequence.clear();
    m_futureStateSequence.clear();
    m_neighborsStateSequence.clear();

    m_pastStateSequence.resize(numSteps);
    m_futureStateSequence.resize(numSteps);
    m_neighborsStateSequence.resize(numSteps);

    for (size_t t = 0; t < numSteps; t++) {
        m_pastStateSequence[t] = state;
        if (async) { asyncUpdateState(getSize()); }
        else { syncUpdateState(); }
        m_futureStateSequence[t] = state;
        m_neighborsStateSequence[t] = getNeighborsState(state);
    }
};

const State Dynamics::getRandomState() {
    size_t N = m_randomGraph.getSize();
    State rnd_state(N);
    uniform_int_distribution<int> dist(0, m_numStates - 1);

    for (size_t i = 0 ; i < N ; i ++){
        rnd_state[i] = dist(m_rng);
    }

    return rnd_state;
};

const NeighborsState Dynamics::getNeighborsState(const State& state) const {
    size_t N = m_randomGraph.getSize();
    NeighborsState neighborSates(N);
    int neighborIdx, edgeMultiplicity;

    for ( auto idx: getGraph() ){
        neighborSates[idx].resize(m_numStates);
        for ( auto neighbor: getGraph().getNeighboursOfIdx(idx) ){
            neighborIdx = neighbor.first;
            edgeMultiplicity = neighbor.second;
            neighborSates[ idx ][ state[neighborIdx] ] += edgeMultiplicity;
        }
    }
    return neighborSates;
};

const VertexNeighborhoodStateSequence Dynamics::getVertexNeighborsState ( const VertexIndex& vertex_idx ) const {
    int numSteps = m_pastStateSequence.size();
    VertexNeighborhoodStateSequence neighborState(numSteps);
    for (auto t=0; t<numSteps; t++) {
        neighborState[t] = m_neighborsStateSequence[t][vertex_idx];
    }
    return neighborState;
};

void Dynamics::updateNeighborStateInPlace(
    VertexIndex vertex_idx,
    VertexState prevVertexState,
    VertexState newVertexState,
    NeighborsState& neighborState) const {
    int neighborIdx, edgeMultiplicity;
    for ( auto neighbor: getGraph().getNeighboursOfIdx(vertex_idx) ){
        neighborIdx = neighbor.first;
        edgeMultiplicity = neighbor.second;
        neighborState[neighborIdx][prevVertexState] -= edgeMultiplicity;
        neighborState[neighborIdx][newVertexState] += edgeMultiplicity;
    }
};

void Dynamics::syncUpdateState(){
    State future_state(m_state);
    NeighborsState neighborState = getNeighborsState(m_state);
    vector<double> transProbs(m_numStates);

    for (auto idx: getGraph()){
        transProbs = getTransitionProbs(m_state[idx], neighborState[idx]);
        future_state[idx] = generateCategorical(transProbs, m_rng);
    }
};

void Dynamics::asyncUpdateState(int numUpdates){
    size_t N = m_randomGraph.getSize();
    VertexState newVertexState;
    State currentState(m_state);
    NeighborsState neighborState = getNeighborsState(m_state);
    vector<double> transProbs(m_numStates);
    uniform_int_distribution<VertexIndex> idxGenerator(0, N-1);

    for (auto i=0; i < numUpdates; i++){
        VertexIndex idx = idxGenerator(m_rng);
        transProbs = getTransitionProbs(currentState[idx], neighborState[idx]);
        newVertexState = generateCategorical(transProbs, m_rng);
        updateNeighborStateInPlace(idx, currentState[idx], newVertexState, neighborState);
        currentState[idx] = newVertexState;
    }
};

double Dynamics::getLogLikelihood() const {
    double log_likelihood = 0;
    vector<int> neighborState(getNumStates(), 0);
    int neighborIdx, edgeMultiplicity;
    for (size_t t = 0; t < m_pastStateSequence.size(); t++){
        for (auto idx: getGraph()){
            fill(neighborState.begin(), neighborState.end(), 0);
            for (auto neighbor: getGraph().getNeighboursOfIdx(idx)){
                neighborIdx = neighbor.first;
                edgeMultiplicity = neighbor.second;
                neighborState[m_pastStateSequence[t][idx]] += edgeMultiplicity;
            }
            log_likelihood += log(getTransitionProb(m_pastStateSequence[t][idx],
                m_futureStateSequence[t][idx], neighborState));
        }
    }
    return log_likelihood;
};

const std::vector<double> Dynamics::getTransitionProbs(VertexState prevVertexState, VertexNeighborhoodState neighborhoodState) const{
    std::vector<double> transProbs(getNumStates());
    for (VertexState nextVertexState = 0; nextVertexState < getNumStates(); nextVertexState++) {
        transProbs[nextVertexState] = getTransitionProb(prevVertexState, nextVertexState, neighborhoodState);
    }
    return transProbs;
};

void Dynamics::updateNeighborStateMapFromEdgeMove(Edge edge, int direction, map<VertexIndex, VertexNeighborhoodStateSequence>& prevNeighborMap, map<VertexIndex, VertexNeighborhoodStateSequence>& nextNeighborMap) const{
    int numSteps = m_pastStateSequence.size();
    VertexIndex v = edge.first, u = edge.second;
    if (prevNeighborMap.count(v) == 0){
        prevNeighborMap[v] = getVertexNeighborsState(v) ;
        nextNeighborMap[v] = getVertexNeighborsState(v) ;
    }
    else if (prevNeighborMap.count(u) == 0){
        prevNeighborMap[u] = getVertexNeighborsState(u);
        nextNeighborMap[u] = getVertexNeighborsState(u) ;
    }

    VertexState vState, uState;
    for (size_t t = 0; t < numSteps; t++) {
        uState = m_pastStateSequence[t][u];
        vState = m_pastStateSequence[t][v];
        nextNeighborMap[u][t][vState] += direction;
        nextNeighborMap[v][t][uState] += direction;
    }
};

double Dynamics::getLogJointRatio(const GraphMove& move) const{
    int numSteps = m_pastStateSequence.size();
    double logLikelihoodRatio = 0;
    set<size_t> verticesAffected;
    map<VertexIndex,VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;

    size_t v, u;
    for (const auto& edge : move.edgesAdded){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }
    for (const auto& edge : move.edgesRemoved){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < numSteps; t++) {
            logLikelihoodRatio += log(getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], nextNeighborMap[idx][t]));
            logLikelihoodRatio -= log(getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], prevNeighborMap[idx][t]));
        }
    }

    return logLikelihoodRatio;
}

void Dynamics::applyMove(const GraphMove& move){
    int numSteps = m_pastStateSequence.size();
    set<VertexIndex> verticesAffected;
    map<VertexIndex, VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;

    VertexNeighborhoodStateSequence neighborState(numSteps);
    size_t v, u;
    for (const auto& edge : move.edgesAdded){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }
    for (const auto& edge : move.edgesRemoved){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < numSteps; t++) {
            m_neighborsStateSequence[t][idx] = nextNeighborMap[idx][t];
        }
    }
    m_randomGraph.applyMove(move);
};

void Dynamics::doMetropolisHastingsStep(double beta, double sample_graph_prior){
    uniform_real_distribution<double> uniform_01(0., 1.);
    double dS = 0;
    if ( sample_graph_prior < uniform_01(m_rng) ){
        m_randomGraph.doMetropolisHastingsStep();
    }
    else{
        GraphMove move = m_randomGraph.proposeMove();
        dS += beta * getLogJointRatio(move) + m_randomGraph.getLogJointRatio(move);
        if ( exp(dS) > uniform_01(m_rng) ){
            applyMove(move);
            m_randomGraph.applyMove(move);
        }
    }
};


}
