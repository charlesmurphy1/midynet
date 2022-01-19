#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"

namespace FastMIDyNet{


// class TestSingleEdgeUniformProposer: public::testing::Test {
// public:
//     FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
//     FastMIDyNet::SingleEdgeUniformProposer proposer;
//     void SetUp() {
//         proposer.setUp(graph);
//     }
// };
//
// TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_addEdge_return0) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{}, {edge}};
//     proposer.updateProbabilities(move);
//     EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
// }
//
// TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity3_return0) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{edge}, {}};
//     proposer.updateProbabilities(move);
//     EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
// }
//
// TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
//     BaseGraph::Edge edge = {0, 3};
//     FastMIDyNet::GraphMove move = {{edge}, {}};
//     proposer.updateProbabilities(move);
//     EXPECT_EQ(proposer.getLogProposalProbRatio(move), -log(.5));
// }
//
// class TestSingleEdgeDegreeProposer: public::testing::Test {
// public:
//     FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
//     FastMIDyNet::SingleEdgeUniformProposer proposer;
//     void SetUp() {
//         proposer.setUp(graph);
//     }
// };
//
// TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_addEdge_return0) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{}, {edge}};
//     proposer.updateProbabilities(move);
//     EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
// }
//
// TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity3_return0) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{edge}, {}};
//     proposer.updateProbabilities(move);
// }
//
// TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
//     BaseGraph::Edge edge = {0, 3};
//     FastMIDyNet::GraphMove move = {{edge}, {}};
//     proposer.updateProbabilities(move);
// }

}
