#include "gtest/gtest.h"

#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"
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
            proposer.setUp(randomGraph);
        }
};

TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_addEdge_return0) {
    FastMIDyNet::GraphMove move = {{}, {inexistentEdge}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), -log(0.5));

    move = {{}, {singleEdge}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity2_return0) {
    FastMIDyNet::GraphMove move = {{doubleEdge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
    FastMIDyNet::GraphMove move = {{singleEdge}, {}};
    proposer.updateProbabilities(move);
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), -log(.5));
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
            proposer.setUp(randomGraph);
        }
};

TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_addEdge_return0) {
    FastMIDyNet::GraphMove move = {{}, {inexistentEdge}};
    proposer.updateProbabilities(move);
}

TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity2_return0) {
    FastMIDyNet::GraphMove move = {{doubleEdge}, {}};
    proposer.updateProbabilities(move);
}

TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio) {
    FastMIDyNet::GraphMove move = {{singleEdge}, {}};
    proposer.updateProbabilities(move);
}

class TestLabeledSingleEdgeUniformProposer: public::testing::Test {
public:
    size_t N = 10, E = 10, B = 3;
    BlockCountDeltaPrior blockCount = {B};
    BlockUniformHyperPrior blocks = {N, blockCount};
    EdgeCountDeltaPrior edgeCount = {E};
    EdgeMatrixUniformPrior edgeMatrix = {edgeCount, blocks};
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(N, blocks, edgeMatrix);
    LabeledSingleEdgeUniformProposer proposer = LabeledSingleEdgeUniformProposer();
    MultiGraph graph;

    void SetUp() {
        randomGraph.sample();
        graph = randomGraph.getGraph();
        proposer.setUp(randomGraph);

    }
};

TEST_F(TestLabeledSingleEdgeUniformProposer, proposerLabelPair_returnValideLabelPair){
    std::pair<BlockIndex, BlockIndex> labelPair = proposer.proposeLabelPair();
    EXPECT_TRUE(labelPair.first < 3);
    EXPECT_TRUE(labelPair.second < 3);
}

// TEST_F(TestLabeledSingleEdgeUniformProposer, proposeMove_withLabelPair_returnEdgeWithCorrectLabels){
//     GraphMove move = proposer.proposeRawMove();
//
//     std::pair<BlockIndex, BlockIndex> lastSampledLabelPair = proposer.getLastSampledLabelPair();
//
//     for (auto edge : move.addedEdges){
//         BlockIndex r = randomGraph.getBlockOfIdx(edge.first), s = randomGraph.getBlockOfIdx(edge.second);
//         if (r > s){
//             size_t tmp = r;
//             r = s;
//             s = tmp;
//         }
//
//         EXPECT_EQ(lastSampledLabelPair.first, r);
//         EXPECT_EQ(lastSampledLabelPair.second, s);
//     }
//     // move.display();
}

}
