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
    m_neighborsState = computeNeighborsState(m_state);

    m_pastStateSequence.clear();
    m_futureStateSequence.clear();
    m_neighborsPastStateSequence.clear();

    for (size_t t = 0; t < m_numSteps; t++) {
        m_pastStateSequence.push_back(m_state);
        m_neighborsPastStateSequence.push_back(m_neighborsState);
        if (async) { asyncUpdateState(getSize()); }
        else { syncUpdateState(); }
        m_futureStateSequence.push_back(m_state);
    }

    computeVertexNeighborsStateMap();
};

const State Dynamics::getRandomState() const{
    size_t N = m_randomGraphPtr->getSize();
    State rnd_state(N);
    uniform_int_distribution<int> dist(0, m_numStates - 1);

    for (size_t i = 0 ; i < N ; i ++)
        rnd_state[i] = dist(rng);

    return rnd_state;
};

const NeighborsState Dynamics::computeNeighborsState(const State& state) const {
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
        neighborState[t] = m_neighborsPastStateSequence[t][vertexIdx];
    }
    return neighborState;
};

void Dynamics::updateNeighborStateInPlace(
    VertexIndex vertexIdx,
    VertexState prevVertexState,
    VertexState newVertexState,
    NeighborsState& neighborState) const {
    int neighborIdx, edgeMultiplicity;
    if (prevVertexState == newVertexState)
        return;
    for ( auto neighbor: getGraph().getNeighboursOfIdx(vertexIdx) ){
        neighborIdx = neighbor.vertexIndex;
        edgeMultiplicity = neighbor.label;
        neighborState[neighborIdx][prevVertexState] -= edgeMultiplicity;
        neighborState[neighborIdx][newVertexState] += edgeMultiplicity;
    }
};

void Dynamics::syncUpdateState(){
    State futureState(m_state);
    vector<double> transProbs(m_numStates);
    NeighborsState newNeighborsState = m_neighborsState;

    for (auto idx: getGraph()){
        transProbs = getTransitionProbs(m_state[idx], m_neighborsState[idx]);
        futureState[idx] = generateCategorical<double, size_t>(transProbs);
        updateNeighborStateInPlace(idx, m_state[idx], futureState[idx], newNeighborsState);

    }
    m_state = futureState;
    m_neighborsState = newNeighborsState;
};

void Dynamics::asyncUpdateState(int numUpdates){
    size_t N = m_randomGraphPtr->getSize();
    VertexState newVertexState;
    State currentState(m_state);
    // m_neighborsState = getNeighborsState(m_state);
    vector<double> transProbs(m_numStates);
    uniform_int_distribution<VertexIndex> idxGenerator(0, N-1);

    for (auto i=0; i < numUpdates; i++){
        VertexIndex idx = idxGenerator(rng);
        transProbs = getTransitionProbs(currentState[idx], m_neighborsState[idx]);
        newVertexState = generateCategorical<double, size_t>(transProbs);
        updateNeighborStateInPlace(idx, currentState[idx], newVertexState, m_neighborsState);
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
                m_neighborsPastStateSequence[t][idx]
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
    edge = getOrderedEdge(edge);
    VertexIndex v = edge.first, u = edge.second;

    if (m_randomGraphPtr->getGraph().getEdgeMultiplicityIdx(edge) == 0 and counter < 0)
        throw std::logic_error("Dynamics: Edge ("
                                + std::to_string(edge.first) + ", "
                                + std::to_string(edge.second) + ") "
                                + "with multiplicity 0 cannot be removed.");


    if (prevNeighborMap.count(v) == 0){
        prevNeighborMap.insert({v, m_vertexMapNeighborsPastStateSequence.at(v)}) ;
        nextNeighborMap.insert({v, m_vertexMapNeighborsPastStateSequence.at(v)}) ;
    }
    if (prevNeighborMap.count(u) == 0){
        prevNeighborMap.insert({u, m_vertexMapNeighborsPastStateSequence.at(u)}) ;
        nextNeighborMap.insert({u, m_vertexMapNeighborsPastStateSequence.at(u)}) ;
    }

    VertexState vState, uState;
    for (size_t t = 0; t < m_numSteps; t++) {
        uState = m_pastStateSequence[t][u];
        vState = m_pastStateSequence[t][v];
        if (u == v)
            nextNeighborMap[u][t][uState] += counter;
        else{
            nextNeighborMap[u][t][vState] += counter;
            nextNeighborMap[v][t][uState] += counter;
        }
    }
};

const double Dynamics::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const{
    double logLikelihoodRatio = 0;
    set<size_t> verticesAffected;
    map<VertexIndex,VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;


    for (const auto& edge : move.addedEdges){
        size_t v = edge.first, u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }
    for (const auto& edge : move.removedEdges){
        size_t v = edge.first, u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < m_numSteps; t++) {
            logLikelihoodRatio += log(
                getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], nextNeighborMap[idx][t])
            );
            logLikelihoodRatio -= log(
                getTransitionProb(m_pastStateSequence[t][idx], m_futureStateSequence[t][idx], prevNeighborMap[idx][t])
            );
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
        m_neighborsState[u][m_state[v]] += 1;
        m_neighborsState[v][m_state[u]] += 1;
    }
    for (const auto& edge : move.removedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborStateMapFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
        m_neighborsState[u][m_state[v]] -= 1;
        m_neighborsState[v][m_state[u]] -= 1;
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < m_numSteps; t++) {
            m_neighborsPastStateSequence[t][idx] = nextNeighborMap[idx][t];
            m_vertexMapNeighborsPastStateSequence.at(idx)[t] = nextNeighborMap[idx][t];
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
    if (m_neighborsPastStateSequence.size() == 0)
        throw SafetyError("Dynamics: unsafe graph family since `m_state` is empty.");
}

}
