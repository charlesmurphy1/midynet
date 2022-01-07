#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

class DummySingleEdgeProposer: public SingleEdgeProposer{
private:
    VertexUniformSampler m_vertexSampler = VertexUniformSampler();
public:
    DummySingleEdgeProposer(){ setVertexSampler(m_vertexSampler); }
};

class TestSingleEdgeProposer: public::testing::Test {
public:
    FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
    FastMIDyNet::DummySingleEdgeProposer proposer;
    void SetUp() {
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

}
