#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"


class TestDoubleEdgeSwap: public::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::DoubleEdgeSwap swapProposer;
        void SetUp() {
            swapProposer.setup(graph);
        }
};


TEST_F(TestDoubleEdgeSwap, setup_anyGraph_samplableSetContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), swapProposer.getSamplableSet().total_weight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), swapProposer.getSamplableSet().size());
}

TEST_F(TestDoubleEdgeSwap, setup_anyGraph_samplableSetHasOnlyOrderedEdges) {
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.first)
                EXPECT_EQ(round(swapProposer.getSamplableSet().get_weight({vertex, neighbor.first})), neighbor.second);
            else
                EXPECT_EQ(swapProposer.getSamplableSet().count({vertex, neighbor.first}), 0);
}

TEST_F(TestDoubleEdgeSwap, updateProbabilities_addExistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{}, {edge}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
}

TEST_F(TestDoubleEdgeSwap, updateProbabilities_addInexistentMultiEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    FastMIDyNet::GraphMove move = {{}, {edge, edge}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
}

TEST_F(TestDoubleEdgeSwap, updateProbabilities_addInexistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    BaseGraph::Edge reversedEdge = {1, 0};
    FastMIDyNet::GraphMove move = {{}, {edge}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
    EXPECT_EQ(swapProposer.getSamplableSet().count(reversedEdge), 0);
}

TEST_F(TestDoubleEdgeSwap, updateProbabilities_removeEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge}, {}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
}

TEST_F(TestDoubleEdgeSwap, updateProbabilities_removeMultiEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge, edge}, {}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-2);
}

TEST_F(TestDoubleEdgeSwap, updateProbabilities_removeAllEdges_edgeRemovedFromSamplableSet) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge, edge, edge}, {}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().count(edge), 0);
}
