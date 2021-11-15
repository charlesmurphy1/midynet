#ifndef FASTMIDYNET_DYNAMICS_FIXTURES_HPP
#define FASTMIDYNET_DYNAMICS_FIXTURES_HPP

#include "gtest/gtest.h"

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/proposer/edge_proposer.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"



namespace FastMIDyNet{

class DummyEdgeProposer: public EdgeProposer{
    public:
        GraphMove operator()() const { return GraphMove(); };
        double getProposalProb(const GraphMove&) { return 0.; };
        void applyMove(const GraphMove&) { };
};

class DummyRandomGraph: public RandomGraph{
    public:
        DummyRandomGraph(size_t size, RNG& rng):
        m_edge_proposer(), RandomGraph(size, m_edge_proposer, rng) {} ;

    void sampleState() { };
    double getLogLikelihood(const MultiGraph&) const { return 0; };
    double getLogJointRatio(const GraphMove& move) const { return 0; };

    private:
        DummyEdgeProposer m_edge_proposer;
};

class DummyDynamics: public Dynamics{
    const double getTransitionProb(
        VertexState prev_vertex_state,
        VertexState next_vertex_state,
        VertexNeighborhoodState neighborhood_state
    ) const { return 0.; };
};

}

FastMIDyNet::MultiGraph getUndirectedHouseMultiGraph(){
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
    FastMIDyNet::MultiGraph graph = FastMIDyNet::MultiGraph(7);
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

class TestDynamics: public::testing::Test{
    public:
        FastMIDyNet::RNG rng;
        size_t N = 7;
        size_t num_states = 3;
        FastMIDyNet::DummyRandomGraph graph = FastMIDyNet::DummyRandomGraph(N, rng);
        // FastMIDyNet::DummyDynamics dynamics(graph, num_states, rng);
        // void SetUp() {
        //     auto graph = getUndirectedHouseMultiGraph();
        //     FastMIDyNet::State state = {0, 0, 0, 1, 1, 1, 2};
        //     dynamics.setState(state);
        //     dynamics.setGraph(graph);
        // }
};





#endif
