#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/types.h"
#include "../fixtures.hpp"


namespace FastMIDyNet{

class TestGibbsUniformBlockProposer: public::testing::Test {
public:
    double SAMPLE_LABEL_PROB=0.1, LABEL_CREATION_PROB=0.5;
    size_t numSamples = 1000;
    const size_t NUM_VERTICES = 100, NUM_EDGES = 250;
    const bool useHyperPrior = false, canonical = false, stubLabeled = false;

    StochasticBlockModelFamily graphPrior = StochasticBlockModelFamily(100, 250, 3, useHyperPrior, canonical, stubLabeled);
    GibbsUniformBlockProposer proposer = GibbsUniformBlockProposer(SAMPLE_LABEL_PROB, LABEL_CREATION_PROB);
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

TEST_F(TestGibbsUniformBlockProposer, proposeLabelMove_returnValidMove){
    std::vector<double> counts(3, 0);
    // for (size_t i = 0; i < numSamples; i++) {
    //     auto move = proposer.proposeLabelMove(0);
    //     ++counts[move.nextLabel];
    //     EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
    //     EXPECT_TRUE(move.addedLabels == 0);
    // }
    // for (auto c : counts)
    //     EXPECT_NEAR(c / numSamples, 1./graphPrior.getLabelCount(), 1e-1);
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
    const std::set<BlockIndex>& getEmptyLabels() { return m_emptyLabels; }
    const std::set<BlockIndex>& getAvailableLabels() { return m_availableLabels; }

    void printAvails(){
        std::cout << "avails: ";
        for (auto k : getAvailableLabels())
            std::cout << k << ", ";
        std::cout << std::endl;
    }

    void printEmpties(){
        std::cout << "empties: ";
        for (auto k : getEmptyLabels())
            std::cout << k << ", ";
        std::cout << std::endl;
    }
};

class TestRestrictedUniformBlockProposer: public::testing::Test {
public:
    double SAMPLE_LABEL_PROB=0.1;
    size_t numSamples = 10;
    const size_t NUM_VERTICES = 100, NUM_EDGES = 250;
    const bool useHyperPrior = true, canonical = false, stubLabeled = false;

    StochasticBlockModelFamily graphPrior = StochasticBlockModelFamily(100, 250, 3, useHyperPrior, canonical, stubLabeled);
    StochasticBlockModelFamily smallGraphPrior = StochasticBlockModelFamily(5, 5, 3, useHyperPrior, canonical, stubLabeled);
    DummyRestrictedUniformBlockProposer proposer = DummyRestrictedUniformBlockProposer(SAMPLE_LABEL_PROB);
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        smallGraphPrior.sample();
        proposer.setUpWithPrior(graphPrior);
        proposer.checkSafety();
    }
    void TearDown(){
        proposer.checkConsistency();
    }
};

TEST_F(TestRestrictedUniformBlockProposer, proposeLabelMove_returnValidMove){
    std::vector<double> counts(3, 0);

    // for (size_t i = 0; i < numSamples; i++) {
    //     auto move = proposer.proposeLabelMove(0);
    //     ++counts[move.nextLabel];
    //     EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
    //     EXPECT_TRUE(move.addedLabels <= 0);
    // }
    // for (auto c : counts)
    //     EXPECT_NEAR(c / numSamples, 1./graphPrior.getLabelCount(), 1e-1);
}

TEST_F(TestRestrictedUniformBlockProposer, proposeNewLabelMove_returnValidMove){
    auto move = proposer.proposeNewLabelMove(0);
    if (move.prevLabel != move.nextLabel){
        EXPECT_TRUE(move.addedLabels == 1);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0));
        EXPECT_NE(proposer.getEmptyLabels().count(move.nextLabel), 0);
    }
}

TEST_F(TestRestrictedUniformBlockProposer, proposeLabelMove_forMoveDestroyingLabel_returnValidMove){
    proposer.setUpWithPrior(smallGraphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getVertexCounts().get(move.prevLabel) != 1){
        smallGraphPrior.sample();
        proposer.setUpWithPrior(smallGraphPrior);
        move = proposer.proposeLabelMove(0);
    }
    EXPECT_EQ(move.prevLabel, smallGraphPrior.getLabelOfIdx(0));
    EXPECT_EQ(move.addedLabels, (move.prevLabel != move.nextLabel) ? -1 : 0);
}

TEST_F(TestRestrictedUniformBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(proposer.getAvailableLabels().size())) ;
    }
}

TEST_F(TestRestrictedUniformBlockProposer, getLogProposalProb_forBlockMoveAddingLabel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        if (move.prevLabel != move.nextLabel)
            EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB)) ;
    }
}

TEST_F(TestRestrictedUniformBlockProposer, getLogProposalProb_forBlockMoveDestroyingLabel_returnCorrectProb){
    proposer.setUpWithPrior(smallGraphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getVertexCounts().get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        smallGraphPrior.sample();
        proposer.setUpWithPrior(smallGraphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, false);
    EXPECT_EQ(logProb, log(1 - SAMPLE_LABEL_PROB) - log(proposer.getAvailableLabels().size())) ;
}

TEST_F(TestRestrictedUniformBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, true);
        if (move.addedLabels == 0)
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
    proposer.setUpWithPrior(smallGraphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getVertexCounts().get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        smallGraphPrior.sample();
        proposer.setUpWithPrior(smallGraphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, true);
    EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB)) ;
}

TEST_F(TestRestrictedUniformBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    auto move = proposer.proposeLabelMove(0);
    proposer.applyLabelMove(move);
    if (move.addedLabels == 0){
        EXPECT_EQ(empties, proposer.getEmptyLabels());
        EXPECT_EQ(avails, proposer.getAvailableLabels());
    }
}

TEST_F(TestRestrictedUniformBlockProposer, applyLabelMove_forBlockMoveAddingLabel){
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    auto move = proposer.proposeNewLabelMove(0);
    proposer.applyLabelMove(move);
    if (empties.size() > 1) EXPECT_NE(empties, proposer.getEmptyLabels()); else EXPECT_EQ(empties, proposer.getEmptyLabels());
    if (move.prevLabel != move.nextLabel)
        EXPECT_NE(avails, proposer.getAvailableLabels());
}

TEST_F(TestRestrictedUniformBlockProposer, applyLabelMove_forBlockMoveDestroyingLabel){
    proposer.setUpWithPrior(smallGraphPrior);
    auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    auto move = proposer.proposeLabelMove(0);
    while(smallGraphPrior.getVertexCounts().get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        smallGraphPrior.sample();
        proposer.setUpWithPrior(smallGraphPrior);
        empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
        move = proposer.proposeLabelMove(0);
    }
    proposer.applyLabelMove(move);
    EXPECT_NE(empties, proposer.getEmptyLabels());
    EXPECT_EQ(proposer.getAvailableLabels().count(move.prevLabel), 0);
}

}
