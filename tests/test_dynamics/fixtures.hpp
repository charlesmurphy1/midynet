#ifndef FASTMIDYNET_DYNAMICS_FIXTURES_HPP
#define FASTMIDYNET_DYNAMICS_FIXTURES_HPP

#include <iostream>
#include "gtest/gtest.h"

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/proposer/edge_proposer.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/undirected_multigraph.h"



namespace FastMIDyNet{

class DummyEdgeProposer: public EdgeProposer{
    public:
        GraphMove operator()() { return GraphMove(); };
        double getProposalProb(const GraphMove&) const { return 0.; };
        void updateProbabilities(const GraphMove&) { };
};

class DummyRandomGraph: public RandomGraph{
    public:
        DummyRandomGraph(size_t size, RNG& rng):
        m_edgeProposer(), RandomGraph(size, m_edgeProposer, rng) {} ;

        void sampleState() { };
        double getLogLikelihood(const MultiGraph&) const { return 0; };
        double getLogJointRatio(const GraphMove& move) const { return 0; };

    private:
        DummyEdgeProposer m_edgeProposer;
};

class DummyDynamics: public Dynamics{
    public:
        DummyDynamics(RandomGraph& randomGraph, int numStates, RNG& rng):
            Dynamics(randomGraph, numStates, rng) { }

        double getTransitionProb(
            VertexState prevVertexState,
            VertexState nextVertexState,
            VertexNeighborhoodState vertexNeighborhoodState
        ) const { return 1. / getNumStates(); }
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
        FastMIDyNet::RNG rng;
        FastMIDyNet::DummyRandomGraph graph = FastMIDyNet::DummyRandomGraph(7, rng);
        FastMIDyNet::DummyDynamics dynamics = FastMIDyNet::DummyDynamics(graph, StateNumber, rng);
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
