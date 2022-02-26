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
    DummyDoubleEdgeSwapProposer swapProposer;
    void SetUp() {
        swapProposer.setUpFromGraph(graph);
    }
};


TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplerContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), swapProposer.getEdgeSampler().getTotalWeight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), swapProposer.getEdgeSampler().getSize());
}

TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplerHasOnlyOrderedEdges) {
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                EXPECT_EQ(round(swapProposer.getEdgeSampler().getEdgeWeight({vertex, neighbor.vertexIndex})), neighbor.label);
            else
                EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight({vertex, neighbor.vertexIndex}), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addExistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{}, {edge}};
    swapProposer.applyGraphMove(move);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addInexistentMultiEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    GraphMove move = {{}, {edge, edge}};
    swapProposer.applyGraphMove(move);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addInexistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    BaseGraph::Edge reversedEdge = {1, 0};
    GraphMove move = {{}, {edge}};
    swapProposer.applyGraphMove(move);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(reversedEdge), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge}, {}};
    swapProposer.applyGraphMove(move);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeMultiEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge, edge}, {}};
    swapProposer.applyGraphMove(move);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)-2);
}

TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeAllEdges_edgeRemovedFromSamplableSet) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{edge, edge, edge}, {}};
    swapProposer.applyGraphMove(move);
    EXPECT_EQ(swapProposer.getEdgeSampler().getEdgeWeight(edge), 0);
}

TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forSomeGraphMove_returnCorrectValue) {
    auto move = swapProposer.proposeMove();
    auto reversedMove = swapProposer.getReverseMove(move);
    double weight = swapProposer.getLogProposalWeight(move);
    swapProposer.applyGraphMove(move);
    double weightAfterMove = swapProposer.getLogProposalWeight(reversedMove);
    swapProposer.applyGraphMove(reversedMove);
    EXPECT_FLOAT_EQ(weightAfterMove - weight, swapProposer.getLogProposalProbRatio(move));
}


}
