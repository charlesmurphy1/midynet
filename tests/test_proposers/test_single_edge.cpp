#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"


class TestSingleEdgeProposer: public::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::SingleEdgeProposer proposer;
        FastMIDyNet::VertexUniformSampler vertexSampler = FastMIDyNet::VertexUniformSampler();
        void SetUp() {
            proposer.setVertexSampler(vertexSampler);
            proposer.setUp(graph);
        }
};

TEST_F(TestSingleEdgeProposer, getLogProposalProbRatio_addEdge_return0) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{}, {edge}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity3_return0) {
    BaseGraph::Edge edge = {0, 2};
    FastMIDyNet::GraphMove move = {{edge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
    BaseGraph::Edge edge = {0, 3};
    FastMIDyNet::GraphMove move = {{edge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), -log(.5));
}
