#include "gtest/gtest.h"

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge/single_edge.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

class TestSingleEdgeUniformProposer: public::testing::Test {
    public:
        EdgeCountDeltaPrior edgeCountPrior = {10};
        ErdosRenyiFamily randomGraph = ErdosRenyiFamily(10, edgeCountPrior);
        SingleEdgeUniformProposer proposer;
        MultiGraph graph;
        BaseGraph::Edge inexistentEdge = {0, 1};
        BaseGraph::Edge singleEdge = {0, 2};
        BaseGraph::Edge doubleEdge = {0, 3};
        void SetUp() {
            randomGraph.sample();
            graph = randomGraph.getGraph();
            graph.setEdgeMultiplicityIdx(inexistentEdge, 0);
            graph.setEdgeMultiplicityIdx(singleEdge, 1);
            graph.setEdgeMultiplicityIdx(doubleEdge, 2);
            randomGraph.setGraph(graph);
            proposer.setUp(graph);
            proposer.checkSafety();
        }
        void TearDown() {
            proposer.checkConsistency();
        }
};

TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_addEdge_return0) {
    FastMIDyNet::GraphMove move = {{}, {inexistentEdge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), -log(0.5));

    move = {{}, {singleEdge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity2_return0) {
    FastMIDyNet::GraphMove move = {{doubleEdge}, {}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
    FastMIDyNet::GraphMove move = {{singleEdge}, {}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), log(.5));
}

class TestSingleEdgeDegreeProposer: public::testing::Test {
    public:
        EdgeCountDeltaPrior edgeCountPrior = {10};
        ErdosRenyiFamily randomGraph = ErdosRenyiFamily(10, edgeCountPrior);
        SingleEdgeDegreeProposer proposer;
        MultiGraph graph;
        BaseGraph::Edge inexistentEdge = {0, 1};
        BaseGraph::Edge singleEdge = {0, 2};
        BaseGraph::Edge doubleEdge = {0, 3};
        void SetUp() {
            randomGraph.sample();
            graph = randomGraph.getGraph();

            graph.setEdgeMultiplicityIdx(inexistentEdge, 0);
            graph.setEdgeMultiplicityIdx(singleEdge, 1);
            graph.setEdgeMultiplicityIdx(doubleEdge, 2);
            randomGraph.setGraph(graph);
            proposer.setUp(graph);
            proposer.checkSafety();
        }
        void TearDown() {
            proposer.checkConsistency();
        }
};

TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_addEdge_return0) {
    FastMIDyNet::GraphMove move = {{}, {inexistentEdge}};
    proposer.applyGraphMove(move);
}

TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity2_return0) {
    FastMIDyNet::GraphMove move = {{doubleEdge}, {}};
    // proposer.applyGraphMove(move);
}

TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
    FastMIDyNet::GraphMove move = {{singleEdge}, {}};
    // proposer.applyGraphMove(move);
}

}
