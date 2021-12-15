#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

class DummyDoubleEdgeSwapProposer: public DoubleEdgeSwapProposer{
public:
    using DoubleEdgeSwapProposer::DoubleEdgeSwapProposer;
    const sset::SamplableSet<BaseGraph::Edge>& getSamplableSet() { return m_edgeSamplableSet; }
};

class TestDoubleEdgeSwapProposer: public::testing::Test {
public:
    MultiGraph graph = getUndirectedHouseMultiGraph();
    DummyDoubleEdgeSwapProposer swapProposer;
    void SetUp() {
        swapProposer.setUp(graph);
    }
};


TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplableSetContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), swapProposer.getSamplableSet().total_weight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), swapProposer.getSamplableSet().size());
}

TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplableSetHasOnlyOrderedEdges) {
    for (auto vertex: graph)
    for (auto neighbor: graph.getNeighboursOfIdx(vertex))
    if (vertex <= neighbor.vertexIndex)
    EXPECT_EQ(round(swapProposer.getSamplableSet().get_weight({vertex, neighbor.vertexIndex})), neighbor.label);
    else
    EXPECT_EQ(swapProposer.getSamplableSet().count({vertex, neighbor.vertexIndex}), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, updateProbabilities_addExistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{}, {edge}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
}

TEST_F(TestDoubleEdgeSwapProposer, updateProbabilities_addInexistentMultiEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    GraphMove move = {{}, {edge, edge}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
}

TEST_F(TestDoubleEdgeSwapProposer, updateProbabilities_addInexistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    BaseGraph::Edge reversedEdge = {1, 0};
    GraphMove move = {{}, {edge}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
    EXPECT_EQ(swapProposer.getSamplableSet().count(reversedEdge), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, updateProbabilities_removeEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge}, {}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
}

TEST_F(TestDoubleEdgeSwapProposer, updateProbabilities_removeMultiEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge, edge}, {}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-2);
}

TEST_F(TestDoubleEdgeSwapProposer, updateProbabilities_removeAllEdges_edgeRemovedFromSamplableSet) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge, edge, edge}, {}};
    swapProposer.updateProbabilities(move);
    EXPECT_EQ(swapProposer.getSamplableSet().count(edge), 0);
}


}
