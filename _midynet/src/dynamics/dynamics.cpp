#include <chrono>
#include <cmath>
#include <map>

#include "BaseGraph/types.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"


using namespace std;
using namespace BaseGraph;


namespace FastMIDyNet {
void Dynamics::sampleState(const State& x0, bool async){
    m_state = x0;

    m_pastStateSequence.clear();
    m_futureStateSequence.clear();
    m_neighborsStateSequence.clear();

    for (size_t t = 0; t < m_numSteps; t++) {
        m_pastStateSequence.push_back(m_state);
        m_neighborsStateSequence.push_back(getNeighborsState(m_state));
        if (async) { asyncUpdateState(getSize()); }
        else { syncUpdateState(); }
        m_futureStateSequence.push_back(m_state);
    }
};

const State Dynamics::getRandomState() const{
    size_t N = m_randomGraphPtr->getSize();
    State rnd_state(N);
    uniform_int_distribution<int> dist(0, m_numStates - 1);

    for (size_t i = 0 ; i < N ; i ++)
        rnd_state[i] = dist(rng);

    return rnd_state;
};

const NeighborsState Dynamics::getNeighborsState(const State& state) const {
    size_t N = m_randomGraphPtr->getSize();
    NeighborsState neighborSates(N);
    int neighborIdx, edgeMultiplicity;

    for ( auto idx: getGraph() ){
        neighborSates[idx].resize(m_numStates);
        for ( auto neighbor: getGraph().getNeighboursOfIdx(idx) ){
            neighborIdx = neighbor.vertexIndex;
            edgeMultiplicity = neighbor.label;
            neighborSates[ idx ][ state[neighborIdx] ] += edgeMultiplicity;
        }
    }
    return neighborSates;
};

const VertexNeighborhoodStateSequence Dynamics::getVertexNeighborsState ( const VertexIndex& vertexIdx ) const {
    VertexNeighborhoodStateSequence neighborState(m_numSteps);
    for (auto t=0; t<m_numSteps; t++) {
        neighborState[t] = m_neighborsStateSequence[t][vertexIdx];
    }
    return neighborState;
};

void Dynamics::updateNeighborStateInPlace(
    VertexIndex vertexIdx,
    VertexState prevVertexState,
    VertexState newVertexState,
    NeighborsState& neighborState) const {
    int neighborIdx, edgeMultiplicity;
    for ( auto neighbor: getGraph().getNeighboursOfIdx(vertexIdx) ){
        neighborIdx = neighbor.vertexIndex;
        edgeMultiplicity = neighbor.label;
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
        future_state[idx] = generateCategorical<double, size_t>(transProbs);
    }
    m_state = future_state;
};

void Dynamics::asyncUpdateState(int numUpdates){
    size_t N = m_randomGraphPtr->getSize();
    VertexState newVertexState;
    State currentState(m_state);
    NeighborsState neighborState = getNeighborsState(m_state);
    vector<double> transProbs(m_numStates);
    uniform_int_distribution<VertexIndex> idxGenerator(0, N-1);

    for (auto i=0; i < numUpdates; i++){
        VertexIndex idx = idxGenerator(rng);
        transProbs = getTransitionProbs(currentState[idx], neighborState[idx]);
        newVertexState = generateCategorical<double, size_t>(transProbs);
        updateNeighborStateInPlace(idx, currentState[idx], newVertexState, neighborState);
        currentState[idx] = newVertexState;
    }
    m_state = currentState;
};

const double Dynamics::getLogLikelihood() const {
    double logLikelihood = 0;
    vector<int> neighborState(getNumStates(), 0);
    int neighborIdx, edgeMultiplicity;
    for (size_t t = 0; t < m_pastStateSequence.size(); t++){
        for (auto idx: getGraph()){
            logLikelihood += log(getTransitionProb(
                m_pastStateSequence[t][idx],
                m_futureStateSequence[t][idx],
                m_neighborsStateSequence[t][idx]
            ));
        }
    }
    return logLikelihood;
};

const std::vector<double> Dynamics::getTransitionProbs(VertexState prevVertexState, VertexNeighborhoodState neighborhoodState) const{
    std::vector<double> transProbs(getNumStates());
    for (VertexState nextVertexState = 0; nextVertexState < getNumStates(); nextVertexState++) {
        transProbs[nextVertexState] = getTransitionProb(prevVertexState, nextVertexState, neighborhoodState);
    }
    return transProbs;
};

void Dynamics::updateNeighborStateMapFromEdgeMove(
    BaseGraph::Edge edge,
    int counter,
    map<VertexIndex, VertexNeighborhoodStateSequence>& prevNeighborMap,
    map<VertexIndex, VertexNeighborhoodStateSequence>& nextNeighborMap) const{
    VertexIndex v = edge.first, u = edge.second;


    if (prevNeighborMap.count(v) == 0){
        prevNeighborMap.insert({v, getVertexNeighborsState(v)}) ;
        nextNeighborMap.insert({v, getVertexNeighborsState(v)}) ;
    }
    if (prevNeighborMap.count(u) == 0){
        prevNeighborMap.insert({u, getVertexNeighborsState(u)}) ;
        nextNeighborMap.insert({u, getVertexNeighborsState(u)}) ;
    }

    VertexState vState, uState;
    for (size_t t = 0; t < m_numSteps; t++) {
        uState = m_pastStateSequence[t][u];
        vState = m_pastStateSequence[t][v];
        nextNeighborMap[u][t][vState] += counter;
        nextNeighborMap[v][t][uState] += counter;
        if (nextNeighborMap[v][t][uState] < 0 or nextNeighborMap[u][t][vState] < 0)
            throw std::logic_error("Negative values!");
    }
};

const double Dynamics::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const{
    double logLikelihoodRatio = 0;
    set<size_t> verticesAffected;
    map<VertexIndex,VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;

    size_t v, u;
    for (const auto& edge : move.addedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }
    for (const auto& edge : move.removedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
    }


    for (const auto& idx: verticesAffected){
        std::string prev, next;
        for (size_t t = 0; t < m_numSteps; t++) {
            logLikelihoodRatio += log(getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], nextNeighborMap[idx][t]));
            prev += std::to_string(getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], prevNeighborMap[idx][t])) + ", ";
            logLikelihoodRatio -= log(getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], prevNeighborMap[idx][t]));
            next += std::to_string(getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], nextNeighborMap[idx][t])) + ", ";
        }
    }

    return logLikelihoodRatio;
}

const double Dynamics::getLogPriorRatioFromGraphMove(const GraphMove& move) const{
    return m_randomGraphPtr->getLogJointRatioFromGraphMove(move);
}

const double Dynamics::getLogJointRatioFromGraphMove(const GraphMove& move) const{
    return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
}


void Dynamics::applyGraphMove(const GraphMove& move){
    set<VertexIndex> verticesAffected;
    map<VertexIndex, VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;
    VertexNeighborhoodStateSequence neighborState(m_numSteps);
    size_t v, u;
    for (const auto& edge : move.addedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }
    for (const auto& edge : move.removedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < m_numSteps; t++) {
            m_neighborsStateSequence[t][idx] = nextNeighborMap[idx][t];
        }
    }
    m_randomGraphPtr->applyGraphMove(move);
}

void Dynamics::checkSafety() const {
    if (m_randomGraphPtr == nullptr)
        throw SafetyError("Dynamics: unsafe graph family since `m_randomGraphPtr` is empty.");
    m_randomGraphPtr->checkSafety();

    if (m_state.size() == 0)
        throw SafetyError("Dynamics: unsafe graph family since `m_state` is empty.");
    if (m_pastStateSequence.size() == 0)
        throw SafetyError("Dynamics: unsafe graph family since `m_state` is empty.");
    if (m_futureStateSequence.size() == 0)
        throw SafetyError("Dynamics: unsafe graph family since `m_state` is empty.");
    if (m_neighborsStateSequence.size() == 0)
        throw SafetyError("Dynamics: unsafe graph family since `m_state` is empty.");
}

}
