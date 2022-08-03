#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>
#include <iostream>

#include "BaseGraph/types.h"

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/dynamics/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"


namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class Dynamics: public NestedRandomVariable{
protected:
    size_t m_numStates;
    size_t m_numSteps;
    State m_state;
    std::vector<State> m_neighborsState;
    const bool m_normalizeCoupling;
    StateSequence m_pastStateSequence;
    StateSequence m_futureStateSequence;
    GraphPriorType* m_graphPriorPtr = nullptr;
    NeighborsStateSequence m_neighborsPastStateSequence;

    void updateNeighborsStateInPlace(
        BaseGraph::VertexIndex vertexIdx,
        VertexState prevVertexState,
        VertexState newVertexState,
        NeighborsState& neighborsState
    ) const ;
    void updateNeighborsStateFromEdgeMove(
        BaseGraph::Edge,
        int direction,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&
    ) const ;

    void checkConsistencyOfNeighborsState() const ;
    void checkConsistencyOfNeighborsPastStateSequence() const ;
public:
    explicit Dynamics(size_t numStates, size_t numSteps, bool normalizeCoupling=true):
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_normalizeCoupling(normalizeCoupling)
        { }
    explicit Dynamics(GraphPriorType& randomGraph, size_t numStates, size_t numSteps, bool normalizeCoupling=true):
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_normalizeCoupling(normalizeCoupling)
        { setGraphPrior(randomGraph); }

    const State& getCurrentState() const { return m_state; }
    const NeighborsState& getCurrentNeighborsState() const { return m_neighborsState; }
    const StateSequence& getPastStates() const { return m_pastStateSequence; }
    const StateSequence& getFutureStates() const { return m_futureStateSequence; }
    const NeighborsStateSequence& getNeighborsPastStates() const { return m_neighborsPastStateSequence; }
    const bool normalizeCoupling() const { return m_normalizeCoupling; }
    void setState(State& state) {
        m_state = state;
        m_neighborsState = computeNeighborsState(m_state);
        #if DEBUG
        checkConsistency();
        #endif
    }
    const MultiGraph& getGraph() const { return m_graphPriorPtr->getState(); }
    void setGraph(const MultiGraph& graph) ;

    const GraphPriorType& getGraphPrior() const { return *m_graphPriorPtr; }
    GraphPriorType& getGraphPriorRef() const { return *m_graphPriorPtr; }
    void setGraphPrior(GraphPriorType& randomGraph) {
        m_graphPriorPtr = &randomGraph;
        m_graphPriorPtr->isRoot(false);
    }

    const size_t getSize() const { return m_graphPriorPtr->getSize(); }
    const size_t getNumStates() const { return m_numStates; }
    const size_t getNumSteps() const { return m_numSteps; }
    void setNumSteps(size_t numSteps) { m_numSteps = numSteps; }

    const State& sample(const State& initialState, bool async=true){
        m_graphPriorPtr->sample();
        sampleState(initialState, async);
        computationFinished();
        return getCurrentState();
    }
    const State& sample(bool async=true){ return sample(getRandomState(), async); }
    void sampleState(const State& initialState, bool async=true);
    void sampleState(bool async=true){ sampleState(getRandomState(), async); }
    void sampleGraph() {
        m_graphPriorPtr->sample();
        setGraph(m_graphPriorPtr->getState());
        computationFinished();
    }
    virtual const State getRandomState() const;
    const NeighborsState computeNeighborsState(const State& state) const;
    const NeighborsStateSequence computeNeighborsStateSequence(const StateSequence& stateSequence) const;

    void syncUpdateState();
    void asyncUpdateState(size_t num_updates);

    const double getLogLikelihood() const;
    const double getLogPrior() const {
        return NestedRandomVariable::processRecursiveFunction<double>([&](){
            return m_graphPriorPtr->getLogJoint();
        }, 0);
    }
    const double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
    virtual const double getTransitionProb(
        VertexState prevVertexState,
        VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const = 0;
    const std::vector<double> getTransitionProbs(
        VertexState prevVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const;

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const;
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const {
        return NestedRandomVariable::processRecursiveConstFunction<double>([&](){
            return m_graphPriorPtr->getLogJointRatioFromGraphMove(move);
        }, 0);
    }
    const double getLogJointRatioFromGraphMove(const GraphMove& move) const {
        return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
    }
    void applyGraphMoveToSelf(const GraphMove& move);
    void applyGraphMove(const GraphMove& move) {
        NestedRandomVariable::processRecursiveFunction([&](){
            applyGraphMoveToSelf(move);
            m_graphPriorPtr->applyGraphMove(move);
        });
    }

    void checkSelfSafety() const override;
    void checkSelfConsistency() const override;

    void computationFinished() const override {
        m_isProcessed = false;
        m_graphPriorPtr->computationFinished();
    }

    bool isSafe() const override {
        return (m_graphPriorPtr != nullptr) and (m_graphPriorPtr->isSafe())
           and (m_state.size() != 0) and (m_pastStateSequence.size() != 0)
           and (m_futureStateSequence.size() != 0) and (m_neighborsPastStateSequence.size() != 0);
    }
};

using PlainDynamics = Dynamics<RandomGraph>;
template<typename Label> using VertexLabeledDynamics = Dynamics<VertexLabeledRandomGraph<Label>>;
using BlockLabeledDynamics = Dynamics<VertexLabeledDynamics<BlockIndex>>;

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::sampleState(const State& x0, bool async){
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

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::setGraph(const MultiGraph& graph) {
    m_graphPriorPtr->setState(graph);
    if (m_pastStateSequence.size() == 0)
        return;
    m_neighborsState = computeNeighborsState(m_state);
    m_neighborsPastStateSequence = computeNeighborsStateSequence(m_pastStateSequence);

    #if DEBUG
    checkConsistency();
    #endif
}

template<typename GraphPriorType>
const State Dynamics<GraphPriorType>::getRandomState() const{
    size_t N = m_graphPriorPtr->getSize();
    State rnd_state(N);
    std::uniform_int_distribution<size_t> dist(0, m_numStates - 1);

    for (size_t i = 0 ; i < N ; i ++)
        rnd_state[i] = dist(rng);

    return rnd_state;
};

template<typename GraphPriorType>
const NeighborsState Dynamics<GraphPriorType>::computeNeighborsState(const State& state) const {
    size_t N = m_graphPriorPtr->getSize();
    NeighborsState neighborsState(N);
    for ( auto idx: getGraph() ){
        neighborsState[idx].resize(m_numStates);
        for ( auto neighbor: getGraph().getNeighboursOfIdx(idx) ){
            neighborsState[ idx ][ state[neighbor.vertexIndex] ] += neighbor.label;
        }
    }
    return neighborsState;
};

template<typename GraphPriorType>
const NeighborsStateSequence Dynamics<GraphPriorType>::computeNeighborsStateSequence(const StateSequence& stateSequence) const {

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


template<typename GraphPriorType>
void Dynamics<GraphPriorType>::updateNeighborsStateInPlace(
    BaseGraph::VertexIndex vertexIdx,
    VertexState prevVertexState,
    VertexState newVertexState,
    NeighborsState& neighborsState) const {
    if (prevVertexState == newVertexState)
        return;
    for ( auto neighbor: getGraph().getNeighboursOfIdx(vertexIdx) ){
        neighborsState[neighbor.vertexIndex][prevVertexState] -= neighbor.label;
        neighborsState[neighbor.vertexIndex][newVertexState] += neighbor.label;
    }
};

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::syncUpdateState(){
    State futureState(m_state);
    std::vector<double> transProbs(m_numStates);

    for (const auto idx: getGraph()){
        transProbs = getTransitionProbs(m_state[idx], m_neighborsState[idx]);
        futureState[idx] = generateCategorical<double, size_t>(transProbs);
    }
    for (const auto idx: getGraph())
        updateNeighborsStateInPlace(idx, m_state[idx], futureState[idx], m_neighborsState);
    m_state = futureState;
};

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::asyncUpdateState(size_t numUpdates){
    size_t N = m_graphPriorPtr->getSize();
    VertexState newVertexState;
    State currentState(m_state);
    std::vector<double> transProbs(m_numStates);
    std::uniform_int_distribution<BaseGraph::VertexIndex> idxGenerator(0, N-1);

    for (auto i=0; i < numUpdates; i++){
        BaseGraph::VertexIndex idx = idxGenerator(rng);
        transProbs = getTransitionProbs(currentState[idx], m_neighborsState[idx]);
        newVertexState = generateCategorical<double, size_t>(transProbs);
        updateNeighborsStateInPlace(idx, currentState[idx], newVertexState, m_neighborsState);
        currentState[idx] = newVertexState;
    }
    m_state = currentState;
};

template<typename GraphPriorType>
const double Dynamics<GraphPriorType>::getLogLikelihood() const {
    double logLikelihood = 0;
    std::vector<int> neighborsState(getNumStates(), 0);
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

template<typename GraphPriorType>
const std::vector<double> Dynamics<GraphPriorType>::getTransitionProbs(VertexState prevVertexState, VertexNeighborhoodState neighborhoodState) const{
    std::vector<double> transProbs(getNumStates());
    for (VertexState nextVertexState = 0; nextVertexState < getNumStates(); nextVertexState++) {
        transProbs[nextVertexState] = getTransitionProb(prevVertexState, nextVertexState, neighborhoodState);
    }
    return transProbs;
};

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::updateNeighborsStateFromEdgeMove(
    BaseGraph::Edge edge,
    int counter,
    std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>& prevNeighborMap,
    std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>& nextNeighborMap) const{
    edge = getOrderedEdge(edge);
    BaseGraph::VertexIndex v = edge.first, u = edge.second;

    if (m_graphPriorPtr->getState().getEdgeMultiplicityIdx(edge) == 0 and counter < 0)
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
        nextNeighborMap[u][t][vState] += counter;
        if (u != v)
            nextNeighborMap[v][t][uState] += counter;
    }
};

template<typename GraphPriorType>
const double Dynamics<GraphPriorType>::getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const{
    double logLikelihoodRatio = 0;
    std::set<size_t> verticesAffected;
    std::map<BaseGraph::VertexIndex,VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;


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


template<typename GraphPriorType>
void Dynamics<GraphPriorType>::applyGraphMoveToSelf(const GraphMove& move) {
    std::set<BaseGraph::VertexIndex> verticesAffected;
    std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;
    VertexNeighborhoodStateSequence neighborsState(m_numSteps);
    size_t v, u;

    for (const auto& edge : move.addedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborsStateFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
        m_neighborsState[u][m_state[v]] += 1;
        if (u != v)
            m_neighborsState[v][m_state[u]] += 1;
    }
    for (const auto& edge : move.removedEdges){
        v = edge.first;
        u = edge.second;
        verticesAffected.insert(v);
        verticesAffected.insert(u);
        updateNeighborsStateFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
        m_neighborsState[u][m_state[v]] -= 1;
        if (u != v)
            m_neighborsState[v][m_state[u]] -= 1;
    }

    for (const auto& idx: verticesAffected){
        for (size_t t = 0; t < m_numSteps; t++) {
            m_neighborsPastStateSequence[idx][t] = nextNeighborMap[idx][t];
        }
    }
}

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::checkConsistencyOfNeighborsPastStateSequence() const {
    if (m_neighborsPastStateSequence.size() == 0)
        return;
    else if (m_neighborsPastStateSequence.size() != getSize())
        throw ConsistencyError(
            "Dynamics",
            "graph prior", "size=" + std::to_string(getSize()),
            "m_neighborsPastStateSequence", "size=" + std::to_string(m_neighborsPastStateSequence.size())
        );
    const auto& actual = m_neighborsPastStateSequence;
    const auto expected = computeNeighborsStateSequence(m_pastStateSequence);
    for (size_t v=0; v<getSize(); ++v){
        if (actual[v].size() != getNumSteps())
            throw ConsistencyError(
                "Dynamics",
                "m_numSteps", "value=" + std::to_string(getNumSteps()),
                "m_neighborsPastStateSequence", "size=" + std::to_string(actual[v].size()),
                "vertex=" + std::to_string(v)
            );
        for (size_t t=0; t<getSize(); ++t){
            if (actual[v][t].size() != getNumStates())
                throw ConsistencyError(
                    "Dynamics",
                    "m_numStates", "value=" + std::to_string(getNumStates()),
                    "m_neighborsPastStateSequence", "size=" + std::to_string(actual[v][t].size()),
                    "vertex=" + std::to_string(v) + ", time=" + std::to_string(t)
                );
            for (size_t s=0; s<m_numStates; ++s){
                if (actual[v][t][s] != expected[v][t][s])
                    throw ConsistencyError(
                        "Dynamics",
                        "neighbor counts", "value=" + std::to_string(expected[v][t][s]),
                        "m_neighborsPastStateSequence", "value=" + std::to_string(actual[v][t][s]),
                        "vertex=" + std::to_string(v) + ", time=" + std::to_string(t) + ", state=" + std::to_string(s)
                    );
            }
        }
    }
}

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::checkConsistencyOfNeighborsState() const {
    const auto& actual = m_neighborsState;
    const auto expected = computeNeighborsState(m_state);
    if (m_neighborsState.size() == 0)
        return;
    else if (actual.size() != getSize())
        throw ConsistencyError(
            "Dynamics",
            "graph prior", "size=" + std::to_string(getSize()),
            "m_neighborsState", "value=" + std::to_string(actual.size())
        );
    for (size_t v=0; v<getSize(); ++v){
        if (actual[v].size() != getNumStates())
            throw ConsistencyError(
                "Dynamics",
                "m_numStates", "value=" + std::to_string(getNumStates()),
                "m_neighborsState", "size=" + std::to_string(actual[v].size()),
                "vertex=" + std::to_string(v)

            );
        for (size_t s=0; s<m_numStates; ++s){
            if (actual[v][s] != expected[v][s])
                throw ConsistencyError(
                    "Dynamics",
                    "neighbor counts", "value=" + std::to_string(expected[v][s]),
                    "m_neighborsState", "value=" + std::to_string(actual[v][s]),
                    "vertex=" + std::to_string(v) + ", state=" + std::to_string(s)
                );
        }
    }

}

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::checkSelfConsistency() const {
    checkConsistencyOfNeighborsPastStateSequence();
    checkConsistencyOfNeighborsState();
}

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::checkSelfSafety() const {
    if (m_graphPriorPtr == nullptr)
        throw SafetyError("Dynamics", "m_graphPriorPtr");
    m_graphPriorPtr->checkSafety();

    if (m_state.size() == 0)
        throw SafetyError("Dynamics", "m_state.size()", "0");
    if (m_pastStateSequence.size() == 0)
        throw SafetyError("Dynamics", "m_pastStateSequence.size()", "0");
    if (m_futureStateSequence.size() == 0)
        throw SafetyError("Dynamics", "m_futureStateSequence.size()", "0");
    if (m_neighborsPastStateSequence.size() == 0)
        throw SafetyError("Dynamics", "m_neighborsPastStateSequence.size()", "0");
}

} // namespace FastMIDyNet

#endif
