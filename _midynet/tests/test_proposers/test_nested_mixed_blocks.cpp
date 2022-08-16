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
        std::cout << "avails: " << std::endl;
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
        std::cout << "empties: " << std::endl;
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
    const size_t SIZE=10, EDGECOUNT=20;
    const bool canonical = false, stubLabeled = false;
    double SAMPLE_LABEL_PROB=0.1, SHIFT=1000;
    size_t numSamples = 10;
    NestedStochasticBlockModelFamily graphPrior = NestedStochasticBlockModelFamily(
        SIZE, EDGECOUNT, canonical, stubLabeled
    );
    DummyRestrictedMixedNestedBlockProposer proposer = DummyRestrictedMixedNestedBlockProposer(SAMPLE_LABEL_PROB, SHIFT);
    void SetUp(){
        seedWithTime();
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        proposer.checkSafety();
    }
    void TearDown(){
        proposer.checkConsistency();
    }


    double getGroundTruthMoveProb(
        BaseGraph::VertexIndex vertex,
        BlockIndex s,
        const MultiGraph& graph,
        const BlockSequence& blocks,
        const LabelGraph& labelGraph,
        size_t blockCount){
            double weight, degree;

            for (const auto& neighbor : graph.getNeighboursOfIdx(vertex)){
                BlockIndex t = blocks[neighbor.vertexIndex];
                size_t m = neighbor.label;
                if (vertex == neighbor.vertexIndex)
                    m *= 2;
                size_t Est = 0;
                if (s < labelGraph.getSize())
                    Est = labelGraph.getEdgeMultiplicityIdx(s, t);
                if (s == t)
                    Est *= 2;
                size_t Et = labelGraph.getDegreeOfIdx(t);
                weight += m * (Est + SHIFT) / (Et + SHIFT * blockCount);
                degree += m;
            }
            if (degree == 0)
                return 1. / blockCount;
            return weight / degree;
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
        proposer.setUpWithNestedPrior(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    EXPECT_EQ(move.prevLabel, graphPrior.getLabelOfIdx(0, move.level));
    EXPECT_EQ(move.addedLabels, (move.prevLabel != move.nextLabel) ? -1 : 0);
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        double logProb = proposer.getLogProposalProb(move, false);
        EXPECT_LE(logProb, 0) ;
    }
}


TEST_F(TestRestrictedMixedNestedBlockProposer, getLogProposalProb_forSomeLabelMove_returnSampledValue){
    CounterMap<std::pair<BlockIndex, Level>> counter;
    size_t numSamples = 100000;
    double tol = 1e-2;
    size_t depth = 3;
    while(graphPrior.getDepth() != depth){
        graphPrior.sample();
    }

    proposer.setUpWithNestedPrior(graphPrior);
    // displayMatrix(graphPrior.getNestedLabels(), "b", true);
    // proposer.printAvails();
    // proposer.printEmpties();
    for (size_t i=0; i<numSamples; ++i){
        // std::cout << "here1" << std::endl;
        BlockMove move = proposer.proposeMove(0);
        // std::cout << "here2" << std::endl;
        counter.increment({move.nextLabel, move.level});
    }

    for (auto s : counter){
        Level level = s.first.second;
        BlockIndex nextLabel = s.first.first, prevLabel = graphPrior.getLabelOfIdx(0, level);
        int dB = 0 ;
        if (graphPrior.getNestedVertexCounts(level).get(nextLabel) == 0)
            dB = 1;
        else if (graphPrior.getNestedVertexCounts(level).get(prevLabel) == 1 and prevLabel != nextLabel)
            dB = -1;

        BlockMove move = {0, prevLabel, nextLabel, dB, level};
        std::cout << move << std::endl;
        double expected = exp(proposer.getLogProposalProb(move));
        double actual = (double)s.second / (double)numSamples;

        double truth;
        if (dB == 1)
            truth = 0.1;
        else
            truth = 0.9 * getGroundTruthMoveProb(
                graphPrior.getLabelOfIdx(0, level - 1),
                nextLabel,
                graphPrior.getNestedLabelGraph(level-1),
                graphPrior.getNestedLabels(level),
                graphPrior.getNestedLabelGraph(level),
                proposer.getAvailableLabels()[level].size()
            );

        truth /= depth;

        // std::cout << " -> expected=" << expected << " actual=" << actual << " truth=" << truth <<  std::endl;

        if (prevLabel != nextLabel){
            EXPECT_NEAR(expected, actual, tol);
            EXPECT_NEAR(expected, truth, 1e-6);
        }
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
    proposer.setUpWithNestedPrior(graphPrior);
    auto move = proposer.proposeLabelMove(0);
    while(graphPrior.getNestedVertexCounts(move.level).get(move.prevLabel) != 1 or move.prevLabel == move.nextLabel){
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        move = proposer.proposeLabelMove(0);
    }
    double logProb = proposer.getLogProposalProb(move, false);
    EXPECT_LE(logProb, 0) ;
}

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forStandardBlockMove_returnCorrectProb){
    for (size_t i = 0; i < numSamples; i++) {
        graphPrior.sample();
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeLabelMove(0);
        while (move.addedLabels != 0 or move.prevLabel == move.nextLabel or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
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
        proposer.setUpWithNestedPrior(graphPrior);
        BlockMove move = proposer.proposeNewLabelMove(0);
        while (move.addedLabels != 1 or move.level == graphPrior.getDepth() - 1 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
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
        proposer.setUpWithNestedPrior(graphPrior);
        auto move = proposer.proposeNewLabelMove(0);
        while (move.addedLabels != 1 or move.level != graphPrior.getDepth() - 1 or not graphPrior.isValidLabelMove(move)){
            graphPrior.sample();
            proposer.setUpWithNestedPrior(graphPrior);
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

TEST_F(TestRestrictedMixedNestedBlockProposer, getLogReverseProposalProb_forBlockMoveDestroyingLabelInLastLevel_returnCorrectProb){
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

        graphPrior.applyLabelMove(move);
        proposer.applyLabelMove(move);

        BlockMove reversedMove = {move.vertexIndex, move.nextLabel, move.prevLabel, -move.addedLabels, move.level};
        double expectedLogProb = proposer.getLogProposalProb(reversedMove);
        EXPECT_NEAR(actualLogProb, expectedLogProb, 1e-6);

    }
}

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forStandardBlockMove_doNothing){
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

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveAddingLabelNotInLastLevel){
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

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveAddingLabelInLastLevel){
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

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveDestroyingLabelNotInLastLevel){
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

TEST_F(TestRestrictedMixedNestedBlockProposer, applyLabelMove_forBlockMoveDestroyingLabelInLastLevel){
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
