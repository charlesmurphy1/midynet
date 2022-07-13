#include "gtest/gtest.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/types.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

class TestGibbsUniformBlockProposer: public::testing::Test {
public:
    double SAMPLE_LABEL_PROB=0.1, LABEL_CREATION_PROB=0.5;
    size_t numSamples = 1000;
    DummySBMGraph graphPrior;
    GibbsUniformBlockProposer proposer = GibbsUniformBlockProposer(SAMPLE_LABEL_PROB, LABEL_CREATION_PROB);
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        proposer.setUp(graphPrior);
        proposer.checkSafety();
    }
    void TearDown(){
        proposer.checkConsistency();
    }
};

TEST_F(TestGibbsUniformBlockProposer, proposeLabelMove_returnValidMove){
    std::vector<double> counts(3, 0);
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        ++counts[move.nextLabel];
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_TRUE(move.addedLabels == 0);
    }
    for (auto c : counts)
        EXPECT_NEAR(c / numSamples, 1./graphPrior.getLabelCount(), 1e-1);
}

TEST_F(TestGibbsUniformBlockProposer, proposeNewLabelMove_returnValidMove){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_TRUE(move.addedLabels != 0);
    }
}

TEST_F(TestGibbsUniformBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getLabelCount())) ;
    }
}

TEST_F(TestGibbsUniformBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, true);
        EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getLabelCount()) + move.addedLabels) ;
    }
}

TEST_F(TestGibbsUniformBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
    auto move = proposer.proposeLabelMove(0);
    proposer.applyLabelMove(move);
}

TEST_F(TestGibbsUniformBlockProposer, getLogProposalProb_forBlockMoveChangingBlockCount_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB)) ;
    }
}

TEST_F(TestGibbsUniformBlockProposer, getLogReverseProposalProb_forBlockMoveChangingBlockCount_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, true);
        EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB)) ;
    }
}

TEST_F(TestGibbsUniformBlockProposer, applyLabelMove_forBlockMoveChangingBlockCount_doNothing){
    auto move = proposer.proposeNewLabelMove(0);
    proposer.applyLabelMove(move);
}


class DummyRestrictedUniformBlockProposer: public RestrictedUniformBlockProposer{
public:
    using RestrictedUniformBlockProposer::RestrictedUniformBlockProposer;
    const std::set<BlockIndex>& getEmptylabels() { return m_emptyLabels; }
    const std::set<BlockIndex>& getAvailableLabels() { return m_availableLabels; }
};

class TestRestrictedUniformBlockProposer: public::testing::Test {
public:
    double SAMPLE_LABEL_PROB=0.1;
    size_t numSamples = 1000;
    DummyRestrictedSBMGraph graphPrior;
    DummyRestrictedSBMGraph smallGraphPrior = DummyRestrictedSBMGraph(5);
    DummyRestrictedUniformBlockProposer proposer = DummyRestrictedUniformBlockProposer(SAMPLE_LABEL_PROB);
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        smallGraphPrior.sample();
        proposer.setUp(graphPrior);
        proposer.checkSafety();
    }
    void TearDown(){
        proposer.checkConsistency();
    }
};

TEST_F(TestRestrictedUniformBlockProposer, proposeLabelMove_returnValidMove){
    std::vector<double> counts(3, 0);
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        ++counts[move.nextLabel];
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_TRUE(move.addedLabels == 0);
    }
    for (auto c : counts)
        EXPECT_NEAR(c / numSamples, 1./graphPrior.getLabelCount(), 1e-1);
}

TEST_F(TestRestrictedUniformBlockProposer, proposeNewLabelMove_returnValidMove){
    auto move = proposer.proposeNewLabelMove(0);
    EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
    EXPECT_EQ(move.nextLabel, graphPrior.getLabelCount());
    EXPECT_TRUE(move.addedLabels == 1);
}

TEST_F(TestRestrictedUniformBlockProposer, proposeLabelMove_forMoveDestroyingLabel_returnValidMove){
    proposer.setUp(smallGraphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getLabelCounts().get(move.prevLabel) != 1){
        smallGraphPrior.sample();
        proposer.setUp(smallGraphPrior);
        move = proposer.proposeLabelMove(0);
    }
    EXPECT_EQ(move.prevLabel, smallGraphPrior.getLabelOfIdx(0));
    EXPECT_EQ(move.addedLabels, (move.prevLabel != move.nextLabel) ? -1 : 0);
}

TEST_F(TestRestrictedUniformBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getLabelCount())) ;
    }
}

TEST_F(TestRestrictedUniformBlockProposer, getLogProposalProb_forBlockMoveAddingLabel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB)) ;
    }
}

TEST_F(TestRestrictedUniformBlockProposer, getLogProposalProb_forBlockMoveDestroyingLabel_returnCorrectProb){
    proposer.setUp(smallGraphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getLabelCounts().get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        smallGraphPrior.sample();
        proposer.setUp(smallGraphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, false);
    EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(proposer.getAvailableLabels().size())) ;
}

TEST_F(TestRestrictedUniformBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, true);
        EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(proposer.getAvailableLabels().size())) ;
    }
}

TEST_F(TestRestrictedUniformBlockProposer, getLogReverseProposalProb_forBlockMoveAddingLabel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, true);
        EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(proposer.getAvailableLabels().size() + move.addedLabels)) ;
    }
}

TEST_F(TestRestrictedUniformBlockProposer, getLogReverseProposalProb_forBlockMoveDestroyingLabel_returnCorrectProb){
    proposer.setUp(smallGraphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getLabelCounts().get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        smallGraphPrior.sample();
        proposer.setUp(smallGraphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, true);
    EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB)) ;
}

TEST_F(TestRestrictedUniformBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
    const auto empties = proposer.getEmptylabels(), avails = proposer.getAvailableLabels();
    auto move = proposer.proposeLabelMove(0);
    proposer.applyLabelMove(move);
    EXPECT_EQ(empties, proposer.getEmptylabels());
    EXPECT_EQ(avails, proposer.getAvailableLabels());
}

TEST_F(TestRestrictedUniformBlockProposer, applyLabelMove_forBlockMoveAddingLabel){
    const auto empties = proposer.getEmptylabels(), avails = proposer.getAvailableLabels();
    auto move = proposer.proposeNewLabelMove(0);
    proposer.applyLabelMove(move);
    if (empties.size() > 1) EXPECT_NE(empties, proposer.getEmptylabels()); else EXPECT_EQ(empties, proposer.getEmptylabels());
    EXPECT_NE(avails, proposer.getAvailableLabels());
}

TEST_F(TestRestrictedUniformBlockProposer, applyLabelMove_forBlockMoveDestroyingLabel){
    proposer.setUp(smallGraphPrior);
    auto empties = proposer.getEmptylabels(), avails = proposer.getAvailableLabels();
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getLabelCounts().get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        smallGraphPrior.sample();
        proposer.setUp(smallGraphPrior);
        empties = proposer.getEmptylabels(), avails = proposer.getAvailableLabels();
        move = proposer.proposeLabelMove(0);
    }
    proposer.applyLabelMove(move);
    EXPECT_NE(empties, proposer.getEmptylabels());
    EXPECT_EQ(avails, proposer.getAvailableLabels());
}

}
