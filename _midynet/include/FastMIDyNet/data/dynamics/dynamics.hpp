#ifndef FAST_MIDYNET_DYNAMICS_H
#define FAST_MIDYNET_DYNAMICS_H


#include <vector>
#include <map>
#include <memory>
#include <iostream>

#include "BaseGraph/types.h"

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"

#include "FastMIDyNet/data/data_model.hpp"
#include "FastMIDyNet/data/types.h"


namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class Dynamics: public DataModel<GraphPriorType>{
protected:
    size_t m_numStates;
    size_t m_numSteps;
    bool m_async;
    std::vector<VertexState> m_state;
    Matrix<VertexState> m_neighborsState;
    const bool m_normalizeCoupling;
    Matrix<VertexState> m_pastStateSequence;
    Matrix<VertexState> m_futureStateSequence;
    Matrix<std::vector<VertexState>> m_neighborsPastStateSequence;
    using BaseClass = DataModel<GraphPriorType>;

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
    void computeConsistentState() override ;
public:
    explicit Dynamics(size_t numStates, size_t numSteps, bool async=false, bool normalizeCoupling=true):
        BaseClass(),
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_async(async),
        m_normalizeCoupling(normalizeCoupling)
        { }
    explicit Dynamics(GraphPriorType& graphPrior, size_t numStates, size_t numSteps, bool async=false, bool normalizeCoupling=true):
        BaseClass(graphPrior),
        m_numStates(numStates),
        m_numSteps(numSteps),
        m_async(async),
        m_normalizeCoupling(normalizeCoupling)
        { }

    const std::vector<VertexState>& getState() const { return m_state; }
    void setState(std::vector<VertexState>& state) {
        m_state = state;
        computeConsistentState();
        #if DEBUG
        checkConsistency();
        #endif
    }
    const Matrix<VertexState>& getNeighborsState() const { return m_neighborsState; }
    const Matrix<VertexState>& getPastStates() const { return m_pastStateSequence; }
    const Matrix<VertexState>& getFutureStates() const { return m_futureStateSequence; }
    const Matrix<std::vector<VertexState>>& getNeighborsPastStates() const { return m_neighborsPastStateSequence; }
    const bool normalizeCoupling() const { return m_normalizeCoupling; }
    const size_t getNumStates() const { return m_numStates; }
    const size_t getNumSteps() const { return m_numSteps; }
    void setNumSteps(size_t numSteps) { m_numSteps = numSteps; }

    void sampleState(const std::vector<VertexState>& initialState);
    void sampleState() override { sampleState(getRandomState()); }
    virtual const State getRandomState() const;
    const NeighborsState computeNeighborsState(const State& state) const;
    const NeighborsStateSequence computeNeighborsStateSequence(const StateSequence& stateSequence) const;

    void syncUpdateState();
    void asyncUpdateState(size_t num_updates);

    const double getLogLikelihood() const override ;
    virtual const double getTransitionProb(
        const VertexState& prevVertexState, const VertexState& nextVertexState, const VertexNeighborhoodState& neighborhoodState
    ) const = 0;
    const std::vector<double> getTransitionProbs(
        const VertexState& prevVertexState,
        const VertexNeighborhoodState& neighborhoodState
    ) const;

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override ;
    void applyGraphMoveToSelf(const GraphMove& move) override;
    void checkSelfSafety() const override;
    void checkSelfConsistency() const override;

    bool isSafe() const override {
        return BaseClass::isSafe()
           and (m_state.size() != 0) and (m_pastStateSequence.size() != 0)
           and (m_futureStateSequence.size() != 0) and (m_neighborsPastStateSequence.size() != 0);
    }
};

using PlainDynamics = Dynamics<RandomGraph>;
template<typename Label> using VertexLabeledDynamics = Dynamics<VertexLabeledRandomGraph<Label>>;
using BlockLabeledDynamics = Dynamics<VertexLabeledDynamics<BlockIndex>>;


template<typename GraphPriorType>
void Dynamics<GraphPriorType>::computeConsistentState() {
    if (m_state.size() != 0)
        m_neighborsState = computeNeighborsState(m_state);
    if (m_pastStateSequence.size() != 0)
        m_neighborsPastStateSequence = computeNeighborsStateSequence(m_pastStateSequence);
}

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::sampleState(const State& x0){
    m_state = x0;
    m_neighborsState = computeNeighborsState(m_state);

    StateSequence reversedPastState;
    StateSequence reversedFutureState;
    NeighborsStateSequence reversedNeighborsPastState;
    for (size_t t = 0; t < m_numSteps; t++) {
        reversedPastState.push_back(m_state);
        reversedNeighborsPastState.push_back(m_neighborsState);
        if (m_async) { asyncUpdateState(BaseClass::getSize()); }
        else { syncUpdateState(); }
        reversedFutureState.push_back(m_state);
    }

    const auto N = BaseClass::getSize();
    const auto& graph = BaseClass::getGraph();
    m_pastStateSequence.clear();
    m_pastStateSequence.resize(N);
    m_futureStateSequence.clear();
    m_futureStateSequence.resize(N);
    m_neighborsPastStateSequence.clear();
    m_neighborsPastStateSequence.resize(N);
    for (const auto& idx : graph){
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
const State Dynamics<GraphPriorType>::getRandomState() const{
    size_t N = BaseClass::getSize();
    State rnd_state(N);
    std::uniform_int_distribution<size_t> dist(0, m_numStates - 1);

    for (size_t i = 0 ; i < N ; i ++)
        rnd_state[i] = dist(rng);

    return rnd_state;
};

template<typename GraphPriorType>
const NeighborsState Dynamics<GraphPriorType>::computeNeighborsState(const State& state) const {
    const auto N = BaseClass::getSize();
    const auto& graph = BaseClass::getGraph();
    NeighborsState neighborsState(N);
    for ( auto idx: graph ){
        neighborsState[idx].resize(m_numStates);
        for ( auto neighbor: graph.getNeighboursOfIdx(idx) ){
            neighborsState[ idx ][ state[neighbor.vertexIndex] ] += neighbor.label;
        }
    }
    return neighborsState;
};

template<typename GraphPriorType>
const NeighborsStateSequence Dynamics<GraphPriorType>::computeNeighborsStateSequence(const StateSequence& stateSequence) const {

    const auto N = BaseClass::getSize();
    const auto& graph = BaseClass::getGraph();
    NeighborsStateSequence neighborsStateSequence(N);
    for ( const auto& idx: graph ){
        neighborsStateSequence[idx].resize(m_numSteps);
        for (size_t t=0; t<m_numSteps; t++){
            neighborsStateSequence[idx][t].resize(m_numStates);
            for ( const auto& neighbor: graph.getNeighboursOfIdx(idx) )
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
    const auto& graph = BaseClass::getGraph();
    if (prevVertexState == newVertexState)
        return;
    for ( auto neighbor: graph.getNeighboursOfIdx(vertexIdx) ){
        neighborsState[neighbor.vertexIndex][prevVertexState] -= neighbor.label;
        neighborsState[neighbor.vertexIndex][newVertexState] += neighbor.label;
    }
};

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::syncUpdateState(){
    State futureState(m_state);
    std::vector<double> transProbs(m_numStates);
    const auto& graph = BaseClass::getGraph();

    for (const auto idx: graph){
        transProbs = getTransitionProbs(m_state[idx], m_neighborsState[idx]);
        futureState[idx] = generateCategorical<double, size_t>(transProbs);
    }
    for (const auto idx: graph)
        updateNeighborsStateInPlace(idx, m_state[idx], futureState[idx], m_neighborsState);
    m_state = futureState;
};

template<typename GraphPriorType>
void Dynamics<GraphPriorType>::asyncUpdateState(size_t numUpdates){
    size_t N = BaseClass::getSize();
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
    const auto& graph = BaseClass::getGraph();
    for (size_t t = 0; t < m_numSteps; t++){
        for (auto idx: graph){
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
const std::vector<double> Dynamics<GraphPriorType>::getTransitionProbs(const VertexState& prevVertexState, const VertexNeighborhoodState& neighborhoodState) const{
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
    const auto& graph = BaseClass::getGraph();

    if (graph.getEdgeMultiplicityIdx(edge) == 0 and counter < 0)
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
    const auto N = BaseClass::getSize();
    if (m_neighborsPastStateSequence.size() == 0)
        return;
    else if (m_neighborsPastStateSequence.size() != N)
        throw ConsistencyError(
            "Dynamics",
            "graph prior", "size=" + std::to_string(N),
            "m_neighborsPastStateSequence", "size=" + std::to_string(m_neighborsPastStateSequence.size())
        );
    const auto& actual = m_neighborsPastStateSequence;
    const auto expected = computeNeighborsStateSequence(m_pastStateSequence);
    for (size_t v=0; v<N; ++v){
        if (actual[v].size() != getNumSteps())
            throw ConsistencyError(
                "Dynamics",
                "m_numSteps", "value=" + std::to_string(getNumSteps()),
                "m_neighborsPastStateSequence", "size=" + std::to_string(actual[v].size()),
                "vertex=" + std::to_string(v)
            );
        for (size_t t=0; t<m_numSteps; ++t){
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
    const auto N = BaseClass::getSize();
    const auto& graph = BaseClass::getGraph();
    if (m_neighborsState.size() == 0)
        return;
    else if (actual.size() != N)
        throw ConsistencyError(
            "Dynamics",
            "graph prior", "size=" + std::to_string(N),
            "m_neighborsState", "value=" + std::to_string(actual.size())
        );
    for (size_t v=0; v<N; ++v){
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
    BaseClass::checkSelfSafety();

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
