#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/nested_label/uniform.hpp"
#include "FastMIDyNet/types.h"
#include "../fixtures.hpp"


namespace FastMIDyNet{

class TestGibbsUniformNestedBlockProposer: public::testing::Test {
public:
    const size_t SIZE=10, EDGECOUNT=25;
    const bool canonical = false, stubLabeled = false;
    double SAMPLE_LABEL_PROB=0.1, LABEL_CREATION_PROB=0.5;
    size_t numSamples = 10;
    NestedStochasticBlockModelFamily graphPrior = NestedStochasticBlockModelFamily(
        SIZE, EDGECOUNT, canonical, stubLabeled
    );
    GibbsUniformNestedBlockProposer proposer = GibbsUniformNestedBlockProposer(SAMPLE_LABEL_PROB, LABEL_CREATION_PROB);
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        proposer.checkSafety();
    }
    void TearDown(){
        proposer.checkConsistency();
    }
};

// TEST_F(TestGibbsUniformNestedBlockProposer, proposeLabelMove_returnValidMove){
//     for (size_t i=0; i<numSamples; ++i){
//         auto move = proposer.proposeLabelMove(0);
//         EXPECT_EQ(graphPrior.getLabelOfIdx(move.vertexIndex, move.level), move.prevLabel);
//     }
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, proposeNewLabelMove_returnValidMove){
//     for (size_t i = 0; i < numSamples; i++) {
//         auto move = proposer.proposeNewLabelMove(0);
//         EXPECT_EQ(graphPrior.getLabelOfIdx(move.vertexIndex, move.level), move.prevLabel);
//         EXPECT_TRUE(move.addedLabels != 0);
//     }
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
//     for (size_t i = 0; i < numSamples; i++) {
//         auto move = proposer.proposeLabelMove(0);
//         double logProb = proposer.getLogProposalProb(move, false);
//         EXPECT_NEAR(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getNestedLabelCount(move.level)) - log(graphPrior.getDepth()), 1e-6) ;
//     }
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
//
//     for (size_t i = 0; i < numSamples; i++) {
//         auto move = proposer.proposeLabelMove(0);
//         double logProb = proposer.getLogProposalProb(move, true);
//         EXPECT_NEAR(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getNestedLabelCount(move.level)) - log(graphPrior.getDepth()), 1e-6) ;
//     }
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
//     auto move = proposer.proposeLabelMove(0);
//     proposer.applyLabelMove(move);
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, getLogProposalProb_forBlockMoveChangingBlockCount_returnCorrectProb){
//     for (size_t i = 0; i < numSamples; i++) {
//         auto move = proposer.proposeNewLabelMove(0);
//         double logProb = proposer.getLogProposalProb(move, false);
//         EXPECT_NEAR(logProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()), 1e-6) ;
//     }
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, getLogReverseProposalProb_forBlockMoveChangingBlockCount_returnCorrectProb){
//     for (size_t i = 0; i < numSamples; i++) {
//         auto move = proposer.proposeNewLabelMove(0);
//         double logProb = proposer.getLogProposalProb(move, true);
//         int dL = (move.level == graphPrior.getDepth() - 1 and move.addedLabels==1) ? 1 : 0;
//         EXPECT_NEAR(logProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth() + dL), 1e-6) ;
//     }
// }
//
// TEST_F(TestGibbsUniformNestedBlockProposer, applyLabelMove_forBlockMoveChangingBlockCount_doNothing){
//     auto move = proposer.proposeNewLabelMove(0);
//     proposer.applyLabelMove(move);
// }


class DummyRestrictedUniformNestedBlockProposer: public RestrictedUniformNestedBlockProposer{
public:
    using RestrictedUniformNestedBlockProposer::RestrictedUniformNestedBlockProposer;
    const std::vector<std::set<BlockIndex>>& getEmptyLabels() { return m_emptyLabels; }
    const std::vector<std::set<BlockIndex>>& getAvailableLabels() { return m_availableLabels; }

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

class TestRestrictedUniformNestedBlockProposer: public::testing::Test {
public:
    const size_t SIZE=10, EDGECOUNT=25;
    const bool canonical = false, stubLabeled = false;
    double SAMPLE_LABEL_PROB=0.1, LABEL_CREATION_PROB=0.5;
    size_t numSamples = 10;
    NestedStochasticBlockModelFamily graphPrior = NestedStochasticBlockModelFamily(
        SIZE, EDGECOUNT, canonical, stubLabeled
    );
    DummyRestrictedUniformNestedBlockProposer proposer = DummyRestrictedUniformNestedBlockProposer(SAMPLE_LABEL_PROB);
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        proposer.checkSafety();
    }
    void TearDown(){
        proposer.checkConsistency();
    }
};

TEST_F(TestRestrictedUniformNestedBlockProposer, proposeLabelMove_returnValidMove){
    for (size_t i=0; i<numSamples; ++i){
        auto move = proposer.proposeLabelMove(0);
        EXPECT_EQ(graphPrior.getLabelOfIdx(move.vertexIndex, move.level), move.prevLabel);
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, proposeNewLabelMove_returnValidMove){
    auto move = proposer.proposeNewLabelMove(0);
    if (move.prevLabel != move.nextLabel){
        EXPECT_TRUE(move.addedLabels == 1);
        EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0, move.level));
        EXPECT_NE(proposer.getEmptyLabels()[move.level].count(move.nextLabel), 0);
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, proposeLabelMove_forMoveDestroyingLabel_returnValidMove){
    auto move = proposer.proposeLabelMove(0);
    while(graphPrior.getNestedVertexCounts(move.level).get(move.prevLabel) != 1){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0, move.level));
    EXPECT_EQ(move.addedLabels, (move.prevLabel != move.nextLabel) ? -1 : 0);
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_NEAR(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()) - log(proposer.getAvailableLabels()[move.level].size()), 1e-6) ;
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogProposalProb_forBlockMoveAddingLabel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        auto move = proposer.proposeNewLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        if (move.prevLabel != move.nextLabel)
            EXPECT_EQ(logProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth())) ;
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogProposalProb_forBlockMoveDestroyingLabel_returnCorrectProb){
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(graphPrior.getNestedVertexCounts(move.level).get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, false);
    EXPECT_NEAR(logProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()) - log(proposer.getAvailableLabels()[move.level].size()), 1e-6) ;
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != 0)
            move = proposer.proposeLabelMove(0);
        double actualLogProb = proposer.getLogProposalProb(move, true);
        EXPECT_NEAR(actualLogProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()) - log(proposer.getAvailableLabels()[move.level].size()), 1e-6) ;
        graphPrior.applyLabelMove(move);
        auto reversedMove = move;
        reversedMove.prevLabel = move.nextLabel;
        reversedMove.nextLabel = move.prevLabel;
        double expectedLogProb = proposer.getLogProposalProb(move);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogReverseProposalProb_forBlockMoveAddingLabelNotInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        BlockMove move = proposer.proposeNewLabelMove(0);
        while (move.addedLabels != 1 or move.level == graphPrior.getDepth() - 1){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
            move = proposer.proposeNewLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        EXPECT_NEAR(actualLogProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()) - log(proposer.getAvailableLabels()[move.level].size() + 1), 1e-6) ;

        proposer.applyLabelMove(move);
        graphPrior.applyLabelMove(move);

        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels, move.level};
        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);
    }
}



TEST_F(TestRestrictedUniformNestedBlockProposer, getLogReverseProposalProb_forBlockMoveAddingLabelInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeNewLabelMove(0);
        while (move.addedLabels != 1 or move.level != graphPrior.getDepth() - 1 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
            move = proposer.proposeNewLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        EXPECT_NEAR(actualLogProb, log(1 - SAMPLE_LABEL_PROB) - log(graphPrior.getDepth() + 1) - log(proposer.getAvailableLabels()[move.level].size() + 1), 1e-6) ;

        graphPrior.applyLabelMove(move);
        proposer.applyLabelMove(move);

        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels, move.level};
        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogReverseProposalProb_forBlockMoveDestroyingLabelNotInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != -1 or move.level >= graphPrior.getDepth() - 2 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
            move = proposer.proposeLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        EXPECT_NEAR(actualLogProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth()), 1e-6) ;
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, getLogReverseProposalProb_forBlockMoveDestroyingLabelInLastLevel_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != -1 or move.level != graphPrior.getDepth() - 2 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
            move = proposer.proposeLabelMove(0);
        }
        double actualLogProb = proposer.getLogProposalProb(move, true);
        EXPECT_NEAR(actualLogProb, log(SAMPLE_LABEL_PROB) - log(graphPrior.getDepth() - (int)(graphPrior.getNestedVertexCounts(move.level).size() == 2)), 1e-6) ;
    }
}

TEST_F(TestRestrictedUniformNestedBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
    graphPrior.sample();
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while (
        move.addedLabels != 0 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        move = proposer.proposeLabelMove(0);
    }

    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    proposer.applyLabelMove(move);
    EXPECT_EQ(empties, proposer.getEmptyLabels());
    EXPECT_EQ(avails, proposer.getAvailableLabels());
}

TEST_F(TestRestrictedUniformNestedBlockProposer, applyLabelMove_forBlockMoveAddingLabelNotInLastLevel){
    graphPrior.sample();
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeNewLabelMove(0);
    while (
        move.addedLabels != 1 or
        move.level == graphPrior.getDepth()-1 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        move = proposer.proposeNewLabelMove(0);
    }
    const auto empties = proposer.getEmptyLabels(), avails = proposer.getAvailableLabels();
    proposer.applyLabelMove(move);
    EXPECT_NE(empties, proposer.getEmptyLabels());
    EXPECT_NE(avails, proposer.getAvailableLabels());
    EXPECT_EQ(avails[move.level].size() + 1, proposer.getAvailableLabels()[move.level].size());
}

TEST_F(TestRestrictedUniformNestedBlockProposer, applyLabelMove_forBlockMoveAddingLabelInLastLevel){
    graphPrior.sample();
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeNewLabelMove(0);
    while (
        move.addedLabels != 1 or
        move.level != graphPrior.getDepth()-1 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
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

TEST_F(TestRestrictedUniformNestedBlockProposer, applyLabelMove_forBlockMoveDestroyingLabelNotInLastLevel){
    graphPrior.sample();
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while (
        move.addedLabels != -1 or
        move.level == graphPrior.getDepth()-2 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
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

TEST_F(TestRestrictedUniformNestedBlockProposer, applyLabelMove_forBlockMoveDestroyingLabelInLastLevel){
    graphPrior.sample();
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while (
        move.addedLabels != -1 or
        move.level != graphPrior.getDepth()-2 or
        graphPrior.getNestedVertexCounts(move.level).size() != 2 or
        not graphPrior.isValidLabelMove(move)
    ){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
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
