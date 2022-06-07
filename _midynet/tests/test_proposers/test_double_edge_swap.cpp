#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

class DummyDoubleEdgeSwapProposer: public DoubleEdgeSwapProposer{
public:
    using DoubleEdgeSwapProposer::DoubleEdgeSwapProposer;
    const EdgeSampler& getEdgeSampler() { return m_edgeSampler; }
};

class TestDoubleEdgeSwapProposer: public::testing::Test {
public:
    MultiGraph graph = getUndirectedHouseMultiGraph();
    MultiGraph toyGraph = getToyMultiGraph();
    DummyDoubleEdgeSwapProposer proposer;
    void SetUp() {
        proposer.setUpFromGraph(graph);
        proposer.checkSafety();
    }
    void TearDown() {
        proposer.checkConsistency();
    }
    const MultiGraph getToyMultiGraph() {
        /*
        0 === 1<>
        |     |
        |     |
        2 --- 3<>
        */
        MultiGraph graph(4);

        graph.addEdgeIdx(0, 1);
        graph.addEdgeIdx(0, 1);
        graph.addEdgeIdx(1, 1);
        graph.addEdgeIdx(0, 2);
        graph.addEdgeIdx(1, 3);
        graph.addEdgeIdx(2, 3);
        graph.addEdgeIdx(3, 3);
        return graph;
    }
};


TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplerContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), proposer.getEdgeSampler().getTotalWeight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), proposer.getEdgeSampler().getSize());
}

TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplerHasOnlyOrderedEdges) {
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                EXPECT_EQ(round(proposer.getEdgeSampler().getEdgeWeight({vertex, neighbor.vertexIndex})), neighbor.label);
            else
                EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight({vertex, neighbor.vertexIndex}), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addExistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{}, {edge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addInexistentMultiEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    GraphMove move = {{}, {edge, edge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addInexistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    BaseGraph::Edge reversedEdge = {1, 0};
    GraphMove move = {{}, {edge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(reversedEdge), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge}, {}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeMultiEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge, edge}, {}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)-2);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeAllEdges_edgeRemovedFromSamplableSet) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge, edge, edge}, {}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forNormalGraphMove_returnCorrectValue) {
    proposer.setUpFromGraph(toyGraph);
    GraphMove move;

    move = {{{0, 2}, {1, 3}}, {{0, 1}, {2, 3}}};
    double w02 = toyGraph.getEdgeMultiplicityIdx(0, 2), w13 = toyGraph.getEdgeMultiplicityIdx(1, 3);
    double w01 = toyGraph.getEdgeMultiplicityIdx(0, 1), w23 = toyGraph.getEdgeMultiplicityIdx(2, 3);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w01 + 1) + log(w23 + 1) - log(w02) - log(w13));

    move = {{{0, 2}, {1, 3}}, {{0, 3}, {1, 2}}};
    double w03 = toyGraph.getEdgeMultiplicityIdx(0, 3), w12 = toyGraph.getEdgeMultiplicityIdx(1, 2);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w03 + 1) + log(w12 + 1) - log(w02) - log(w13));
}

TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forDoubleLoopyGraphMove_returnCorrectValue) {
    proposer.setUpFromGraph(toyGraph);
    GraphMove move;

    move = {{{1, 1}, {3, 3}}, {{1, 3}, {1, 3}}};

    double w11 = toyGraph.getEdgeMultiplicityIdx(1, 1), w33 = toyGraph.getEdgeMultiplicityIdx(3, 3);
    double w13 = toyGraph.getEdgeMultiplicityIdx(1, 3);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w13 + 2) + log(w13 + 1) - log(w11) - log(w33) - log(4));

}

TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forSingleLoopyGraphMove_returnCorrectValue) {
    proposer.setUpFromGraph(toyGraph);
    GraphMove move;

    move = {{{1, 1}, {0, 2}}, {{0, 1}, {1, 2}}};
    double w11 = toyGraph.getEdgeMultiplicityIdx(1, 1), w02 = toyGraph.getEdgeMultiplicityIdx(0, 2);
    double w01 = toyGraph.getEdgeMultiplicityIdx(0, 1), w12 = toyGraph.getEdgeMultiplicityIdx(1, 2);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w01 + 1) + log(w12 + 1) - log(w11) - log(w02) - log(2));
}

TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forDoubleEdgeGraphMove_returnCorrectValue) {
    proposer.setUpFromGraph(toyGraph);
    GraphMove move;

    move = {{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}}};
    double w01 = toyGraph.getEdgeMultiplicityIdx(0, 1);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), 0);

    move = {{{0, 1}, {0, 1}}, {{0, 0}, {1, 1}}};
    double w00 = toyGraph.getEdgeMultiplicityIdx(0, 0), w11 = toyGraph.getEdgeMultiplicityIdx(1, 1);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(4) + log(w11 + 1) + log(w00 + 1) - log(w01) - log(w01 - 1));
}

TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forHingeGraphMove_returnCorrectValue) {
    proposer.setUpFromGraph(toyGraph);
    GraphMove move;

    move = {{{1, 3}, {2, 3}}, {{1, 3}, {2, 3}}};
    double w13 = toyGraph.getEdgeMultiplicityIdx(1, 3), w23 = toyGraph.getEdgeMultiplicityIdx(2, 3);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), 0);

    move = {{{1, 3}, {2, 3}}, {{1, 2}, {3, 3}}};
    double w12 = toyGraph.getEdgeMultiplicityIdx(1, 2), w33 = toyGraph.getEdgeMultiplicityIdx(3, 3);
    EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(2) + log(w12 + 1) + log(w33 + 1) - log(w13) - log(w23));
}


}
