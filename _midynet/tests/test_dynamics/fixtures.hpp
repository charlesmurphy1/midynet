#ifndef FASTMIDYNET_DYNAMICS_FIXTURES_HPP
#define FASTMIDYNET_DYNAMICS_FIXTURES_HPP

#include <iostream>
#include "gtest/gtest.h"

#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/dynamics/dynamics.hpp"
#include "FastMIDyNet/types.h"
#include "BaseGraph/undirected_multigraph.h"



namespace FastMIDyNet{

static const size_t NUM_STEPS = 10;

class DummyRandomGraph: public RandomGraph{
    size_t m_edgeCount;
public:
    using RandomGraph::RandomGraph;
    DummyRandomGraph(size_t size): RandomGraph(size) {}

    void setGraph(const MultiGraph& graph) override{
        RandomGraph::setGraph(graph);
        m_edgeCount = graph.getTotalEdgeNumber();
    }

    const size_t& getEdgeCount() const override { return m_edgeCount; }

    void sample() override { };
    const double getLogLikelihood() const override { return 0; }
    const double getLogPrior() const override { return 0; }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override{ return 0; }
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0; }


};

class DummyDynamics: public Dynamics<RandomGraph>{
    public:
        DummyDynamics(RandomGraph& randomGraph, int numStates, size_t numSteps):
            Dynamics(randomGraph, numStates, numSteps) { }

        const double getTransitionProb(
            VertexState prevVertexState,
            VertexState nextVertexState,
            VertexNeighborhoodState vertexNeighborhoodState
        ) const { return 1. / getNumStates(); }

        void updateNeighborsStateFromEdgeMove(
            BaseGraph::Edge edge,
            int direction,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&prev,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&next
        ) const { Dynamics::updateNeighborsStateFromEdgeMove(edge, direction, prev, next); }
};

static FastMIDyNet::MultiGraph getUndirectedHouseMultiGraph(){
    //     /*
    //      * (0)      (1)
    //      * ||| \   / | \
    //      * |||  \ /  |  \
    //      * |||   X   |  (4)
    //      * |||  / \  |  /
    //      * ||| /   \ | /
    //      * (2)------(3)-----(5)
    //      *
    //      *      (6)
    //      */
    // STATE = {0,0,0,1,1,1,2}
    // NEIGHBORS_STATE = {{3, 1, 0}, {1, 2, 0}, {4, 1, 0}, {3, 1, 1}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}}
    FastMIDyNet::MultiGraph graph(7);
    graph.addMultiedgeIdx(0, 2, 3);
    graph.addEdgeIdx(0, 3);
    graph.addEdgeIdx(1, 2);
    graph.addEdgeIdx(1, 3);
    graph.addEdgeIdx(1, 4);
    graph.addEdgeIdx(2, 3);
    graph.addEdgeIdx(3, 4);
    graph.addEdgeIdx(3, 5);

    return graph;

}

template <size_t StateNumber>
class TestDynamics: public::testing::Test{
    public:
        FastMIDyNet::DummyRandomGraph graph = FastMIDyNet::DummyRandomGraph(7);
        FastMIDyNet::DummyDynamics dynamics = FastMIDyNet::DummyDynamics(graph, StateNumber, NUM_STEPS);
        void SetUp() {
            auto graph = getUndirectedHouseMultiGraph();
            FastMIDyNet::State state = {0, 0, 0, 1, 1, 1, 2};
            dynamics.setState(state);
            dynamics.setGraph(graph);
        }
};

using TestBinaryDynamics=TestDynamics<2>;


} // namespace FastMIDyNet


#endif
