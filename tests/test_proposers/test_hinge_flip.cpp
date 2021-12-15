#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"


class TestHingeFlip: public::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::HingeFlip proposer;
        void SetUp() {
            proposer.setUp(graph);
        }
};


TEST_F(TestHingeFlip, setup_anyGraph_edgeSamplableSetContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), proposer.getEdgeSamplableSet().total_weight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), proposer.getEdgeSamplableSet().size());
}


TEST_F(TestHingeFlip, setup_anyGraph_samplableSetHasOnlyOrderedEdges) {
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                EXPECT_EQ(round(proposer.getEdgeSamplableSet().get_weight({vertex, neighbor.vertexIndex})), neighbor.label);
            else
                EXPECT_EQ(proposer.getEdgeSamplableSet().count({vertex, neighbor.vertexIndex}), 0);
}


TEST_F(TestHingeFlip, updateProbabilities_addExistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{}, {edge}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
}


TEST_F(TestHingeFlip, updateProbabilities_addMultiEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    FastMIDyNet::GraphMove move = {{}, {edge, edge}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
}


TEST_F(TestHingeFlip, updateProbabilities_addInexistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    BaseGraph::Edge reversedEdge = {1, 0};
    FastMIDyNet::GraphMove move = {{}, {edge}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
    EXPECT_EQ(proposer.getEdgeSamplableSet().count(reversedEdge), 0);
}


TEST_F(TestHingeFlip, updateProbabilities_removeEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
}


TEST_F(TestHingeFlip, updateProbabilities_removeMultiEdge_edgeWeightDecreased) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge, edge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-2);
}


TEST_F(TestHingeFlip, updateProbabilities_removeAllEdges_edgeRemovedFromSamplableSet) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge, edge, edge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getEdgeSamplableSet().count(edge), 0);
}
