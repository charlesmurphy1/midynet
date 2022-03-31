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

    StateSequence reversedPastState;
    StateSequence reversedFutureState;
    NeighborsStateSequence reversedNeighborsPastState;
    for (size_t t = 0; t < m_numSteps; t++) {
        reversedPastState.push_back(m_state);
        reversedNeighborsPastState.push_back(m_neighborsState);
        if (async) { asyncUpdateState(getSize()); }
        else { syncUpdateState(); }
        reversedFutureState.push_back(m_state);
    }


    m_pastStateSequence.clear();
    m_pastStateSequence.resize(getSize());
    m_futureStateSequence.clear();
    m_futureStateSequence.resize(getSize());
    m_neighborsPastStateSequence.clear();
    m_neighborsPastStateSequence.resize(getSize());
    for (const auto& idx : getGraph()){
        m_pastStateSequence[idx].resize(m_numSteps);
        m_futureStateSequence[idx].resize(m_numSteps);
        m_neighborsPastStateSequence[idx].resize(m_numSteps);
        for (size_t t = 0; t < m_numSteps; t++){
            m_pastStateSequence[idx][t] = reversedPastState[t][idx];
            m_futureStateSequence[idx][t] = reversedFutureState[t][idx];
            m_neighborsPastStateSequence[idx][t] = reversedNeighborsPastState[t][idx];
        }
    }

    #if DEBUG
    checkConsistency();
    #endif

}

void Dynamics::setGraph(const MultiGraph& graph) {
    m_randomGraphPtr->setGraph(graph);
    if (m_pastStateSequence.size() == 0)
        return;
    m_neighborsState = computeNeighborsState(m_state);
    m_neighborsPastStateSequence = computeNeighborsStateSequence(m_pastStateSequence);

    #if DEBUG
    checkConsistency();
    #endif
}

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
    NeighborsState neighborsState(N);
    int neighborIdx, edgeMultiplicity;

    for ( auto idx: getGraph() ){
        neighborsState[idx].resize(m_numStates);
        for ( auto neighbor: getGraph().getNeighboursOfIdx(idx) ){
            neighborIdx = neighbor.vertexIndex;
            edgeMultiplicity = neighbor.label;
            neighborsState[ idx ][ state[neighborIdx] ] += edgeMultiplicity;
        }
    }
    return neighborsState;
};

const NeighborsStateSequence Dynamics::computeNeighborsStateSequence(const StateSequence& stateSequence) const {

    NeighborsStateSequence neighborsStateSequence(getSize());
    for ( const auto& idx: getGraph() ){
        neighborsStateSequence[idx].resize(m_numSteps);
        for (size_t t=0; t<m_numSteps; t++){
            neighborsStateSequence[idx][t].resize(m_numStates);
            for ( const auto& neighbor: getGraph().getNeighboursOfIdx(idx) )
                neighborsStateSequence[ idx ][t][ stateSequence[neighbor.vertexIndex][t] ] += neighbor.label;
        }
    }
    return neighborsStateSequence;
};


void Dynamics::updateNeighborsStateInPlace(
    VertexIndex vertexIdx,
    VertexState prevVertexState,
    VertexState newVertexState,
    NeighborsState& neighborsState) const {
    int neighborIdx, edgeMultiplicity;
    if (prevVertexState == newVertexState)
        return;
    for ( auto neighbor: getGraph().getNeighboursOfIdx(vertexIdx) ){
        neighborIdx = neighbor.vertexIndex;
        edgeMultiplicity = neighbor.label;
        neighborsState[neighborIdx][prevVertexState] -= edgeMultiplicity;
        neighborsState[neighborIdx][newVertexState] += edgeMultiplicity;
    }
};

void Dynamics::syncUpdateState(){
    State futureState(m_state);
    vector<double> transProbs(m_numStates);

    for (const auto idx: getGraph()){
        transProbs = getTransitionProbs(m_state[idx], m_neighborsState[idx]);
        futureState[idx] = generateCategorical<double, size_t>(transProbs);
    }
    for (const auto idx: getGraph())
        updateNeighborsStateInPlace(idx, m_state[idx], futureState[idx], m_neighborsState);
    displayMatrix(m_neighborsState, "n");
    m_state = futureState;
};

void Dynamics::asyncUpdateState(int numUpdates){
    size_t N = m_randomGraphPtr->getSize();
    VertexState newVertexState;
    State currentState(m_state);
    vector<double> transProbs(m_numStates);
    uniform_int_distribution<VertexIndex> idxGenerator(0, N-1);

    for (auto i=0; i < numUpdates; i++){
        VertexIndex idx = idxGenerator(rng);
        transProbs = getTransitionProbs(currentState[idx], m_neighborsState[idx]);
        newVertexState = generateCategorical<double, size_t>(transProbs);
        updateNeighborsStateInPlace(idx, currentState[idx], newVertexState, m_neighborsState);
        currentState[idx] = newVertexState;
    }
    m_state = currentState;
};

const double Dynamics::getLogLikelihood() const {
    double logLikelihood = 0;
    vector<int> neighborsState(getNumStates(), 0);
    int neighborIdx, edgeMultiplicity;
    for (size_t t = 0; t < m_numSteps; t++){
        for (auto idx: getGraph()){
            logLikelihood += log(getTransitionProb(
                m_pastStateSequence[idx][t],
                m_futureStateSequence[idx][t],
                m_neighborsPastStateSequence[idx][t]
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

void Dynamics::updateNeighborsStateFromEdgeMove(
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
        prevNeighborMap.insert({v, m_neighborsPastStateSequence[v]}) ;
        nextNeighborMap.insert({v, m_neighborsPastStateSequence[v]}) ;
    }
    if (prevNeighborMap.count(u) == 0){
        prevNeighborMap.insert({u, m_neighborsPastStateSequence[u]}) ;
        nextNeighborMap.insert({u, m_neighborsPastStateSequence[u]}) ;
    }

    VertexState vState, uState;
    for (size_t t = 0; t < m_numSteps; t++) {
        uState = m_pastStateSequence[u][t];
        vState = m_pastStateSequence[v][t];
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
        updateNeighborsStateFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
    }
    for (const auto& edge : move.removedEdges){
        size_t v = edge.first, u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborsStateFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < m_numSteps; t++) {
            logLikelihoodRatio += log(
                getTransitionProb(m_pastStateSequence[idx][t], m_futureStateSequence[idx][t], nextNeighborMap[idx][t])
            );
            logLikelihoodRatio -= log(
                getTransitionProb(m_pastStateSequence[idx][t], m_futureStateSequence[idx][t], prevNeighborMap[idx][t])
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
    VertexNeighborhoodStateSequence neighborsState(m_numSteps);
    size_t v, u;

    for (const auto& edge : move.addedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborsStateFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
        m_neighborsState[u][m_state[v]] += 1;
        m_neighborsState[v][m_state[u]] += 1;
    }
    for (const auto& edge : move.removedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborsStateFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
        m_neighborsState[u][m_state[v]] -= 1;
        m_neighborsState[v][m_state[u]] -= 1;
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < m_numSteps; t++) {
            m_neighborsPastStateSequence[idx][t] = nextNeighborMap[idx][t];
        }
    }
    m_randomGraphPtr->applyGraphMove(move);
}

void Dynamics::checkConsistencyOfNeighborsPastState() const {
    if (m_neighborsPastStateSequence != computeNeighborsStateSequence(m_pastStateSequence))
        throw ConsistencyError("Dynamics: `m_neighborsPastStateSequence` is inconsistent with past states.");
    if (m_neighborsState != computeNeighborsState(m_state))
        throw ConsistencyError("Dynamics: `m_neighborsState` is inconsistent with `m_state`.");

}

void Dynamics::checkConsistency() const {
    checkConsistencyOfNeighborsPastState();
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
