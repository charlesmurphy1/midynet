#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/nested_label/mixed.hpp"
#include "FastMIDyNet/types.h"
#include "../fixtures.hpp"


namespace FastMIDyNet{


class DummyRestrictedMixedNestedBlockProposer: public RestrictedMixedNestedBlockProposer{
public:
    using RestrictedMixedNestedBlockProposer::RestrictedMixedNestedBlockProposer;
    const std::vector<std::set<BlockIndex>>& getEmptyLabels() { return RestrictedMixedNestedBlockProposer::m_emptyLabels; }
    const std::vector<std::set<BlockIndex>>& getAvailableLabels() { return RestrictedMixedNestedBlockProposer::m_availableLabels; }

    void printAvails(){
        std::cout << "avails: ";
        Level l=0;
        for (const auto& avails : getAvailableLabels()){
            std::cout << "\t Level " << l << ":";
            for (auto k : avails)
                std::cout << " " << k << " ";
            std::cout << std::endl;
            ++l;
        }
        std::cout << std::endl;
    }

    void printEmpties(){
        std::cout << "empties: ";
        Level l=0;
        for (const auto& empties : getEmptyLabels()){
            std::cout << "\t Level " << l << ":";
            for (auto k : empties)
                std::cout << " " << k << " ";
            std::cout << std::endl;
            ++l;
        }
        std::cout << std::endl;
    }
};

class TestRestrictedMixedNestedBlockProposer: public::testing::Test {
public:
    size_t SIZE=10, EDGECOUNT=20;
    double SAMPLE_LABEL_PROB=0.1;
    size_t numSamples = 10;
    DummyNestedSBMGraph graphPrior = DummyNestedSBMGraph(SIZE, EDGECOUNT);
    DummyRestrictedMixedNestedBlockProposer proposer = DummyRestrictedMixedNestedBlockProposer(SAMPLE_LABEL_PROB);
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

TEST_F(TestRestrictedMixedNestedBlockProposer, proposeLabelMove_returnValidMove){
    for (size_t i=0; i<numSamples; ++i){
        auto move = proposer.proposeLabelMove(0);
        EXPECT_EQ(graphPrior.getLabelOfIdx(move.vertexIndex, move.level), move.prevLabel);
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, proposeNewLabelMove_returnValidMove){
    auto move = proposer.proposeNewLabelMove(0);
    if (move.prevLabel != move.nextLabel){
        EXPECT_TRUE(move.addedLabels == 1);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0, move.level));
        EXPECT_NE(proposer.getEmptyLabels()[move.level].count(move.nextLabel), 0);
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, proposeLabelMove_forMoveDestroyingLabel_returnValidMove){
    auto move = proposer.proposeLabelMove(0);
    while(graphPrior.getNestedVertexCounts(move.level).get(move.prevLabel) != 1){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0, move.level));
    EXPECT_EQ(move.addedLabels, (move.prevLabel != move.nextLabel) ? -1 : 0);
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUp(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_LE(logProb, 0) ;
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogProposalProb_forBlockMoveAddingLabel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        if (move.prevLabel != move.nextLabel)
            EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth())) ;
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogProposalProb_forBlockMoveDestroyingLabel_returnCorrectProb){
    proposer.setUp(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(graphPrior.getNestedVertexCounts(move.level).get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, false);
    EXPECT_LE(logProb, 0) ;
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUp(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != 0 or move.prevLabel == move.nextLabel or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUp(graphPrior);
            move = proposer.proposeLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        graphPrior.applyLabelMove(move);
        proposer.applyLabelMove(move);
        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, move.addedLabels, move.level};
        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forBlockMoveAddingLabelNotInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUp(graphPrior);
        BlockMove move = proposer.proposeNewLabelMove(0);
        while (move.addedLabels != 1 or move.level == graphPrior.getDepth() - 1 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUp(graphPrior);
            move = proposer.proposeNewLabelMove(0);
        }

        double actualLogProb = proposer.getLogProposalProb(move, true);

        proposer.applyLabelMove(move);
        graphPrior.applyLabelMove(move);

        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels, move.level};

        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forBlockMoveAddingLabelInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUp(graphPrior);
        auto move = proposer.proposeNewLabelMove(0);
        while (move.addedLabels != 1 or move.level != graphPrior.getDepth() - 1 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUp(graphPrior);
            move = proposer.proposeNewLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);

        graphPrior.applyLabelMove(move);
        proposer.applyLabelMove(move);

        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels, move.level};
        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forBlockMoveDestroyingLabelNotInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUp(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != -1 or move.level >= graphPrior.getDepth() - 2 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUp(graphPrior);
            move = proposer.proposeLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        EXPECT_NEAR(actualLogProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()), 1e-6) ;
    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forBlockMoveDestroyingLabelInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUp(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != -1 or move.level != graphPrior.getDepth() - 2 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUp(graphPrior);
            move = proposer.proposeLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        
        graphPrior.applyLabelMove(move);
        proposer.applyLabelMove(move);

        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels, move.level};
        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);

    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
    graphPrior.sample();
    proposer.setUp(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while (
        move.addedLabels != 0 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeLabelMove(0);
    }

    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    proposer.applyLabelMove(move);
    EXPECT_EQ(empties, proposer.getEmptyLabels());
    EXPECT_EQ(avails, proposer.getAvailableLabels());
}

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveAddingLabelNotInLastLevel){
    graphPrior.sample();
    proposer.setUp(graphPrior);
    auto move = proposer.proposeNewLabelMove(0);
    while (
        move.addedLabels != 1 or
        move.level == graphPrior.getDepth()-1 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeNewLabelMove(0);
    }
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    proposer.applyLabelMove(move);
    EXPECT_EQ(empties, proposer.getEmptyLabels());
    EXPECT_NE(avails, proposer.getAvailableLabels());
    EXPECT_EQ(avails[move.level].size() + 1, proposer.getAvailableLabels()[move.level].size());
}

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveAddingLabelInLastLevel){
    graphPrior.sample();
    proposer.setUp(graphPrior);
    auto move = proposer.proposeNewLabelMove(0);
    while (
        move.addedLabels != 1 or
        move.level != graphPrior.getDepth()-1 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeNewLabelMove(0);
    }
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    proposer.applyLabelMove(move);
    EXPECT_NE(empties, proposer.getEmptyLabels());
    EXPECT_EQ(empties.size() + 1, proposer.getEmptyLabels().size());
    EXPECT_NE(avails, proposer.getAvailableLabels());
    EXPECT_EQ(avails[move.level].size() + 1, proposer.getAvailableLabels()[move.level].size());
    EXPECT_EQ(avails.size() + 1, proposer.getAvailableLabels().size());
}

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveDestroyingLabelNotInLastLevel){
    graphPrior.sample();
    proposer.setUp(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while (
        move.addedLabels != -1 or
        move.level == graphPrior.getDepth()-2 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    graphPrior.applyLabelMove(move);
    proposer.applyLabelMove(move);
    EXPECT_NE(empties, proposer.getEmptyLabels());
    EXPECT_EQ(empties.size(), proposer.getEmptyLabels().size());
    EXPECT_NE(avails, proposer.getAvailableLabels());
    EXPECT_EQ(avails[move.level].size() - 1, proposer.getAvailableLabels()[move.level].size());
    EXPECT_EQ(avails.size(), proposer.getAvailableLabels().size());
}

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveDestroyingLabelInLastLevel){
    graphPrior.sample();
    proposer.setUp(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while (
        move.addedLabels != -1 or
        move.level != graphPrior.getDepth()-2 or
        graphPrior.getNestedVertexCounts(move.level).size() != 2 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUp(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    graphPrior.applyLabelMove(move);
    proposer.applyLabelMove(move);
    EXPECT_NE(empties, proposer.getEmptyLabels());
    EXPECT_EQ(empties.size() - 1, proposer.getEmptyLabels().size());
    EXPECT_NE(avails, proposer.getAvailableLabels());
    EXPECT_EQ(avails[move.level].size() - 1, proposer.getAvailableLabels()[move.level].size());
    EXPECT_EQ(avails.size() - 1, proposer.getAvailableLabels().size());
}

}
