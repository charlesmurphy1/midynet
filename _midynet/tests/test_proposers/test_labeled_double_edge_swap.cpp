#include "gtest/gtest.h"

#include "fixtures.hpp"
#include "FastMIDyNet/proposer/edge_proposer/labeled_double_edge_swap.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class TestLabeledDoubleEdgeSwapProposer: public ::testing::Test{
public:
    DummySBMGraph randomGraph = DummySBMGraph();
    LabeledDoubleEdgeSwapProposer proposer = LabeledDoubleEdgeSwapProposer();
    void SetUp(){
        // seedWithTime();
        randomGraph.sample();
        proposer.setUp(randomGraph);
    }
};

TEST_F(TestLabeledDoubleEdgeSwapProposer, proposeMove){
    auto move = proposer.proposeMove();
    for (auto edge : move.removedEdges)
        EXPECT_GT(randomGraph.getGraph().getEdgeMultiplicityIdx(edge), 0);

    for (auto edge : move.addedEdges)
        EXPECT_GE(randomGraph.getGraph().getEdgeMultiplicityIdx(edge), 0);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, onLabelCreation_doNothing){
    BlockMove move = {0, 0, 0};
    proposer.onLabelCreation(move);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, onLabelDeletion_doNothing){
    BlockMove move = {0, 0, 0};
    proposer.onLabelDeletion(move);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, getLogProposalProbRatio_return0){
    auto move = proposer.proposeMove();
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyGraphMove_forEdgeAdded){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    proposer.applyGraphMove({{}, {{0,1}}});
    EXPECT_EQ(totalEdgeCountBefore + 1, proposer.getTotalEdgeCount());

}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyGraphMove_forSelfLoopAdded){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    proposer.applyGraphMove({{}, {{0,0}}});
    EXPECT_EQ(totalEdgeCountBefore + 1, proposer.getTotalEdgeCount());
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyGraphMove_forSomeGraphMove){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    auto move = proposer.proposeMove();
    proposer.applyGraphMove(move);
    EXPECT_EQ(totalEdgeCountBefore, proposer.getTotalEdgeCount());
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyBlockMove_forSomeBlockMove){

}


}
