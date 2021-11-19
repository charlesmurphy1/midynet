#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge_move.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"


class TestSingleEdgeMove: public::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::SingleEdgeMove edgeMover;
        void SetUp() {
            edgeMover.setup(graph);
        }
};


TEST_F(TestSingleEdgeMove, setup_anyGraph_samplableSetContainsAllVertices) {
    EXPECT_EQ(graph.getSize(), edgeMover.getSamplableSet().total_weight());
    EXPECT_EQ(graph.getSize(), edgeMover.getSamplableSet().size());
}

TEST_F(TestSingleEdgeMove, getLogProposalProbRatio_addEdge_return0) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{}, {edge}};
    edgeMover.updateProbabilities(move);
    EXPECT_EQ(edgeMover.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeMove, getLogProposalProbRatio_removeEdgeWithMultiplicity3_return0) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge}, {}};
    edgeMover.updateProbabilities(move);
    EXPECT_EQ(edgeMover.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeMove, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
    BaseGraph::Edge edge = {0, 3};
    FastMIDyNet::GraphMove move = {{edge}, {}};
    edgeMover.updateProbabilities(move);
    EXPECT_EQ(edgeMover.getLogProposalProbRatio(move), -log(.5));
}
