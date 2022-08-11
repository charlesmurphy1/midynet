#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/label/mixed.hpp"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "../fixtures.hpp"



namespace FastMIDyNet{

class TestGibbsMixedBlockProposer: public::testing::Test{
public:
    double SAMPLE_LABEL_PROB=0.1, LABEL_CREATION_PROB=0.5, SHIFT=1;
    size_t numSamples = 100;
    const size_t NUM_VERTICES = 100, NUM_EDGES = 250;
    const bool useHyperPrior = false, canonical = false, stubLabeled = false;

    StochasticBlockModelFamily graphPrior = StochasticBlockModelFamily(100, 250, 3, useHyperPrior, canonical, stubLabeled);
    GibbsMixedBlockProposer proposer = GibbsMixedBlockProposer(SAMPLE_LABEL_PROB, LABEL_CREATION_PROB, SHIFT);
    bool expectConsistencyError = false;
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        proposer.setUpWithPrior(graphPrior);
        proposer.checkSafety();
    }

    void TearDown(){
        if (not expectConsistencyError)
            proposer.checkConsistency();
    }
};

TEST_F(TestGibbsMixedBlockProposer, proposeLabelMove_returnValidMove){
    for(size_t i=0; i<numSamples; ++i){
        auto move = proposer.proposeLabelMove(0);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_EQ(move.addedLabels, 0);
    }
}

TEST_F(TestGibbsMixedBlockProposer, proposeNewLabelMove_returnValidMove){
    for(size_t i=0; i<numSamples; ++i){
        auto move = proposer.proposeNewLabelMove(0);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_NE(move.addedLabels, 0);
    }
}

TEST_F(TestGibbsMixedBlockProposer, getLogProposalProb_forSomeLabelMove_returnCorrectProb){
    for(size_t i=0; i<10; ++i){
        auto move = proposer.proposeLabelMove(0);
        while (move.prevLabel == move.nextLabel)
            move = proposer.proposeLabelMove(0);
        LabelMove<BlockIndex> reverseMove = {move.vertexIndex, move.nextLabel, move.prevLabel};
        double logProb = proposer.getLogProposalProb(move, false);
        proposer.applyLabelMove(move);
        graphPrior.applyLabelMove(move);
        double revLogProb = proposer.getLogProposalProb(reverseMove, true);
        EXPECT_EQ(logProb, revLogProb);
    }
}

TEST_F(TestGibbsMixedBlockProposer, getLogProposalProb_forLabelMoveAddingNewLabel_returnCorrectProb){
    auto move = proposer.proposeNewLabelMove(0);
    LabelMove<BlockIndex> reverseMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels};
    double logProb = proposer.getLogProposalProb(move, false);
    proposer.applyLabelMove(move);
    graphPrior.applyLabelMove(move);
    double revLogProb = proposer.getLogProposalProb(reverseMove, true);
    EXPECT_EQ(logProb, revLogProb);

}

class TestRestrictedMixedBlockProposer: public::testing::Test{
public:
    double SAMPLE_LABEL_PROB=0.1, LABEL_CREATION_PROB=0.5, SHIFT=1;
    size_t numSamples = 100;
    const size_t NUM_VERTICES = 100, NUM_EDGES = 250;
    const bool useHyperPrior = true, canonical = false, stubLabeled = false;

    StochasticBlockModelFamily graphPrior = StochasticBlockModelFamily(100, 250, 3, useHyperPrior, canonical, stubLabeled);
    RestrictedMixedBlockProposer proposer = RestrictedMixedBlockProposer(SAMPLE_LABEL_PROB, SHIFT);

    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        proposer.setUpWithPrior(graphPrior);
        proposer.checkSafety();
    }

    void TearDown(){
        proposer.checkConsistency();
    }
};

TEST_F(TestRestrictedMixedBlockProposer, proposeLabelMove_returnValidMove){
    for(size_t i=0; i<numSamples; ++i){
        auto move = proposer.proposeLabelMove(0);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_NE(move.addedLabels, 1);
    }
}

TEST_F(TestRestrictedMixedBlockProposer, proposeNewLabelMove_returnValidMove){
    for(size_t i=0; i<numSamples; ++i){
        auto move = proposer.proposeNewLabelMove(0);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        if (move.prevLabel != move.nextLabel)
            EXPECT_EQ(move.addedLabels, 1);
    }
}

TEST_F(TestRestrictedMixedBlockProposer, getLogProposalProb_forSomeLabelMove_returnCorrectProb){
    for(size_t i=0; i<10; ++i){
        auto move = proposer.proposeLabelMove(0);
        while (move.prevLabel == move.nextLabel or graphPrior.getVertexCounts().get(move.prevLabel) == 1)
            move = proposer.proposeLabelMove(0);
        LabelMove<BlockIndex> reverseMove = {move.vertexIndex, move.nextLabel, move.prevLabel};
        double logProb = proposer.getLogProposalProb(move, false);
        proposer.applyLabelMove(move);
        graphPrior.applyLabelMove(move);
        double revLogProb = proposer.getLogProposalProb(reverseMove, true);
        EXPECT_EQ(logProb, revLogProb);
    }
}


TEST_F(TestRestrictedMixedBlockProposer, getLogProposalProb_forLabelMoveAddingNewLabel_returnCorrectProb){
    auto move = proposer.proposeNewLabelMove(0);
    LabelMove<BlockIndex> reverseMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels};
    double logProb = proposer.getLogProposalProb(move, false);
    proposer.applyLabelMove(move);
    graphPrior.applyLabelMove(move);
    double revLogProb = proposer.getLogProposalProb(reverseMove, true);
    EXPECT_EQ(logProb, revLogProb);
}

}
