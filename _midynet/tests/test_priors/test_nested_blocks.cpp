#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "../fixtures.hpp"
#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

using namespace FastMIDyNet;

std::string displayNestedVertexCount(std::vector<CounterMap<BlockIndex>> nestedVertexCount, std::string name="nr", bool display=false){
    size_t l = 0;
    std::stringstream ss;
    ss << name << " = [" << std::endl;
    for (auto nr : nestedVertexCount){
        ss << "\t [ ";
        for (auto k : nr)
            ss << "(" << k.first << "->" << k.second << ") ";
        ss << "]" << std::endl;
        ++l;
    }
    ss << "]";

    if (display)
        std::cout << ss.str() << std::endl;
    return ss.str();

}


class TestNestedBlockPrior: public ::testing::Test {
public:

    size_t GRAPH_SIZE=10, NUM_SAMPLES=100;
    NestedBlockUniformPrior prior = NestedBlockUniformPrior(GRAPH_SIZE);

    bool expectConsistencyError = false;
    void SetUp() {
        prior.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            prior.checkConsistency();
    }

    BlockMove proposeNestedBlockMove(
            BaseGraph::VertexIndex id,
            Level level,
            size_t depth=4,
            bool creatingNewBlock=false,
            bool destroyingBlock=false){
        size_t it = 0;

        while (it < 100){
            prior.sample();
            if (prior.getDepth() != depth)
                continue;
            BlockIndex r, s;
            int addedLabels = 0;
            r = prior.getBlockOfIdx(id, level);
            if (prior.getNestedVertexCounts(level)[r] == 1 and not destroyingBlock)
                continue;
            if (creatingNewBlock){
                addedLabels = 1;
                s = prior.getNestedBlockCount(level);
            } else if (destroyingBlock){
                s = 1;
                addedLabels = -1;
                for (size_t i=0; i<prior.getSize(); ++i){
                    r = prior.getBlockOfIdx(i, level);
                    if (prior.getNestedVertexCounts(level)[r] == 1){
                        id = i;
                        break;
                    }
                    s = r;
                }
            } else{
                s = sampleUniformly(0, (int)prior.getNestedBlockCount(level) - 1);
                if (prior.getNestedVertexCounts(level)[r] == 1)
                    continue;
            }
            BlockMove move = {id, r, s, addedLabels, level};
            if (prior.isValidBlockMove(move) and r != s)
                return move;
            ++it;
        }
        throw std::logic_error("Could not create valid move.");
    }



};

TEST_F(TestNestedBlockPrior, sampleState_returnConsistentState){
    prior.sample();
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, getNestedState){
    prior.sample();
    const auto& nestedBlocks = prior.getNestedState();
    EXPECT_EQ(nestedBlocks.size(), prior.getDepth());
    size_t l=0;
    for (auto b: nestedBlocks){
        EXPECT_TRUE(prior.getMaxBlockCountFromPartition(b) <= prior.getNestedBlockCount(l));
        size_t N = prior.getNestedBlockCount(l - 1);
        EXPECT_EQ(b.size(), N);
        ++l;
    }
}

TEST_F(TestNestedBlockPrior, getNestedStateAtLevel){
    prior.sample();
    for (size_t l=0; l<prior.getDepth(); ++l)
        EXPECT_EQ(prior.getNestedState(l), prior.getNestedState()[l]);
}

TEST_F(TestNestedBlockPrior, getNestedVertexCounts){
    prior.sample();
    const auto& vertexCounts = prior.getNestedVertexCounts();
    EXPECT_EQ(vertexCounts.size(), prior.getDepth());
}


TEST_F(TestNestedBlockPrior, getNestedAbsVertexCounts){
    prior.sample();
    while(prior.getDepth() != 5)
        prior.sample();
    const auto& vertexCounts = prior.getNestedAbsVertexCounts();
}

TEST_F(TestNestedBlockPrior, getBlockOfIdx_forSomeVertexAtAllLevels_returnCorrectIndex){
    prior.sample();
    while(prior.getDepth()!=3)
        prior.sample();
    BaseGraph::VertexIndex vertex = 7;
    BlockIndex index = prior.getNestedState(0)[vertex];
    EXPECT_EQ(prior.getBlockOfIdx(vertex, 0), index);
    index = prior.getNestedState(1)[index];
    EXPECT_EQ(prior.getBlockOfIdx(vertex, 1), index);
    index = prior.getNestedState(2)[index];
    EXPECT_EQ(prior.getBlockOfIdx(vertex, 2), index);
}

TEST_F(TestNestedBlockPrior, creatingNewLevel_forMoveNotCreatingLevel_returnTrue){
    prior.sample();
    BlockMove move = {0, 0, 0, 1, ((int)prior.getDepth())-1};
    EXPECT_TRUE(prior.creatingNewLevel(move));
}

TEST_F(TestNestedBlockPrior, creatingNewLevel_forMoveCreatingLevel_returnFalse){
    prior.sample();
    // BlockMove move = {0, 0, 0, 1, 0};
    // EXPECT_FALSE(prior.creatingNewLevel(move));
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveAtFirstLevel_returnConsistentState){
    BlockMove move = proposeNestedBlockMove(0, 0, 3);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveAtSomeLevel_returnConsistentState){
    BlockMove move = proposeNestedBlockMove(0, 1, 3);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
    EXPECT_NO_THROW(prior.checkConsistency());
}



TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveAddingBlockAtFirstLevelNotCreatingNewLevel_returnConsistentState){
    BlockMove move = proposeNestedBlockMove(0, 0, 3, true);
    std::vector<size_t> prevBlockCounts = prior.getNestedBlockCount();
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
    for (Level l=0; l<prior.getDepth(); ++l){
        size_t expected = prevBlockCounts[l] + ((l == move.level) ? 1 : 0) ;
        EXPECT_EQ(expected, prior.getNestedBlockCount(l));
    }
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveAddingBlockAtSomeLevelNotCreatingNewLevel_returnConsistentState){
    BlockMove move = proposeNestedBlockMove(0, 1, 4, true);
    std::vector<size_t> prevBlockCounts = prior.getNestedBlockCount();
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
    for (Level l=0; l<prior.getDepth(); ++l){
        if (l == move.level)
        EXPECT_EQ(prevBlockCounts[l] + 1, prior.getNestedBlockCount(l));
        else
        EXPECT_EQ(prevBlockCounts[l], prior.getNestedBlockCount(l));
    }
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveAddingBlockCreatingNewLevel_returnConsistentState){

    BlockMove move = proposeNestedBlockMove(0, 2, 4, true);
    std::vector<size_t> prevBlockCounts = prior.getNestedBlockCount();
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
    for (Level l=0; l<prior.getDepth(); ++l){
        if (l == move.level + 1)
            EXPECT_EQ(1, prior.getNestedBlockCount(l));
        else if (l == move.level)
            EXPECT_EQ(prevBlockCounts[l] + 1, prior.getNestedBlockCount(l));
        else
            EXPECT_EQ(prevBlockCounts[l], prior.getNestedBlockCount(l));
    }
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveDestroyingBlock_returnConsistentState){
    BlockMove move = proposeNestedBlockMove(0, 1, 3, false, true);
    prior.applyLabelMove(move);
    EXPECT_NO_THROW(prior.checkConsistency());
}


TEST_F(TestNestedBlockPrior, applyLabelMove_forBlockMoveDestroyingBlockAtSecondToLastLevel_returnConsistentState){
    prior.sample();
    // BlockMove move = proposeNestedBlockMove(0, 2, 3, false, true);
    // std::cout << move.display() << std::endl;
    // displayMatrix(prior.getNestedState(), "b", true);
    // prior.applyLabelMove(move);
    // displayMatrix(prior.getNestedState(), "b", true);
    // EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, isValidBlockMove_forMoveAtFirstLevelNotCreatingLabel_returnCorrectValue){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 0;
    for (size_t i=0; i<NUM_SAMPLES; ++i){
        BlockMove move = proposeNestedBlockMove(vertex, level, 4);
        bool expectedIsValue = (prior.getNestedState(move.level + 1)[move.prevLabel] == prior.getNestedState(move.level + 1)[move.nextLabel]);
        EXPECT_EQ(prior.isValidBlockMove(move), expectedIsValue);
    }
}

TEST_F(TestNestedBlockPrior, isValidBlockMove_forMoveAtSomeLevelNotCreatingLabel_returnCorrectValue){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 1;
    for (size_t i=0; i<NUM_SAMPLES; ++i){
        BlockMove move = proposeNestedBlockMove(vertex, level, 4);

        bool expectedIsValue = (prior.getNestedState(move.level + 1)[move.prevLabel] == prior.getNestedState(move.level + 1)[move.nextLabel]);
        EXPECT_EQ(prior.isValidBlockMove(move), expectedIsValue);
    }
}

TEST_F(TestNestedBlockPrior, reduceHierarchy_forSomeState_returnReduced){

    prior.sample();
    while(prior.getDepth() != 5)
        prior.sample();
    const auto nestedState = prior.getNestedState();
    const auto reducedNestedState = prior.reduceHierarchy(nestedState);

    // displayMatrix(nestedState, "b", true);
    // displayMatrix(reducedNestedState, "[reduced] b", true);

}

// class TestNestedBlockUniformPrior: public ::testing::Test {
//     public:
//
//         size_t GRAPH_SIZE=10, NUM_SAMPLES=100;
//         NestedBlockUniformPrior prior = NestedBlockUniformPrior(GRAPH_SIZE);
//
//         bool expectConsistencyError = false;
//         void SetUp() {
//             prior.checkSafety();
//             prior.sample();
//         }
//         void TearDown(){
//             if (not expectConsistencyError)
//                 prior.checkConsistency();
//         }
// };
//
// TEST_F(TestNestedBlockUniformPrior, setSize_forNewSize_returnStateWithNewSize) {
//     prior.setSize(11);
//     prior.sample();
//     EXPECT_EQ(prior.getState().size(), 11);
// }
//
// TEST_F(TestNestedBlockUniformPrior, getLogLikelihood_returnSomeOfLogLikelihoodAtEachLevel) {
//     double actualLogLikelihood = prior.getLogLikelihood();
//     double expectedLogLikelihood = 0;
//     for (Level l=0; l<prior.getDepth(); ++l)
//         expectedLogLikelihood += prior.getLogLikelihoodAtLevel(l);
//     EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
// }
//
// TEST_F(TestNestedBlockUniformPrior, getLogLikelihoodAtLevel_forAllLevel_returnCorrectValue) {
//     for (Level l=0; l<prior.getDepth(); ++l){
//         double actualLogLikelihood = prior.getLogLikelihoodAtLevel(l);
//         double graphSize = prior.getNestedBlockCount(l - 1);
//         double blockCount = prior.getNestedBlockCount(l);
//         double expectedLogLikelihood = -graphSize * log(blockCount);
//         EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
//     }
// }
//
// TEST_F(TestNestedBlockUniformPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveAtFirstLevel_returnCorrectRatio) {
//     BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
//     Level level = 0;
//     BlockMove move = {0, prior.getBlockOfIdx(0, level), sampleUniformly<BlockIndex>(0, prior.getNestedBlockCount(level) - 1), 0, level};
//     while( not prior.isValidBlockMove(move) ){
//         prior.sample();
//          move = {0, prior.getBlockOfIdx(0, level), sampleUniformly<BlockIndex>(0, prior.getNestedBlockCount(level) - 1), 0, level};
//      }
//     double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = prior.getLogLikelihood();
//     prior.applyLabelMove(move);
//     double logLikelihoodAfter = prior.getLogLikelihood();
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
// }




class TestNestedBlockUniformHyperPrior: public ::testing::Test {
public:

    size_t GRAPH_SIZE=10, NUM_SAMPLES=100;
    NestedBlockUniformHyperPrior prior = NestedBlockUniformHyperPrior(GRAPH_SIZE);

    bool expectConsistencyError = false;
    void SetUp() {
        prior.checkSafety();
        prior.sample();
    }
    void TearDown(){
        if (not expectConsistencyError)
            prior.checkConsistency();
    }

    BlockMove proposeNestedBlockMove(
            BaseGraph::VertexIndex id,
            Level level,
            size_t depth=4,
            bool creatingNewBlock=false,
            bool destroyingBlock=false){
        size_t it = 0;

        while (it < 100){
            prior.sample();
            if (prior.getDepth() != depth)
                continue;
            BlockIndex r, s;
            int addedLabels = 0;
            r = prior.getBlockOfIdx(id, level);
            if (prior.getNestedVertexCounts(level)[r] == 1 and not destroyingBlock)
                continue;
            if (creatingNewBlock){
                addedLabels = 1;
                s = prior.getNestedBlockCount(level);
            } else if (destroyingBlock){
                s = 1;
                addedLabels = -1;
                for (size_t i=0; i<prior.getSize(); ++i){
                    r = prior.getBlockOfIdx(i, level);
                    if (prior.getNestedVertexCounts(level)[r] == 1){
                        id = i;
                        break;
                    }
                    s = r;
                }
            } else{
                s = sampleUniformly(0, (int)prior.getNestedBlockCount(level) - 1);
                if (prior.getNestedVertexCounts(level)[r] == 1)
                    continue;
            }
            BlockMove move = {id, r, s, addedLabels, level};
            if (prior.isValidBlockMove(move) and r != s)
                return move;
            ++it;
        }
        throw std::logic_error("Could not create valid move.");
    }
};

TEST_F(TestNestedBlockUniformHyperPrior, setSize_forNewSize_returnStateWithNewSize) {
    prior.setSize(11);
    prior.sample();
    EXPECT_EQ(prior.getState().size(), 11);
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihood_returnSomeOfLogLikelihoodAtEachLevel) {
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = 0;
    for (Level l=0; l<prior.getDepth(); ++l)
        expectedLogLikelihood += prior.getLogLikelihoodAtLevel(l);
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodAtLevel_forAllLevel_returnCorrectValue) {
    for (Level l=0; l<prior.getDepth(); ++l){
        double actualLogLikelihood = prior.getLogLikelihoodAtLevel(l);
        double graphSize = (l == 0) ? prior.getSize() : prior.getNestedAbsVertexCounts(l - 1).size();
        double blockCount = prior.getNestedAbsVertexCounts(l).size();
        double expectedLogLikelihood = -logBinomialCoefficient(graphSize - 1, blockCount - 1) - logFactorial(graphSize);

        for (const auto& nr : prior.getNestedVertexCounts(l)){
            if (prior.getNestedAbsVertexCounts(l)[nr.first] > 0)
                expectedLogLikelihood += logFactorial(nr.second);
        }

        EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
    }
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveAtFirstLevel_returnCorrectRatio) {
    BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
    Level level = 0;
    BlockMove move = proposeNestedBlockMove(vertex, level, 4);
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveAddingBlockAtFirstLevel_returnCorrectRatio) {
    BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
    Level level = 0;
    BlockMove move = proposeNestedBlockMove(vertex, level, 4, true);
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveDestroyingBlockAtFirstLevel_returnCorrectRatio) {
    BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
    Level level = 0;
    BlockMove move = proposeNestedBlockMove(vertex, level, 4, false, true);
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveCreatingBlockAtLastLevel_returnCorrectRatio) {
    BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
    Level level = 2;
    BlockMove move = proposeNestedBlockMove(vertex, level, 4, true);
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveDestroyingBlockAtLastLevel_returnCorrectRatio) {
    std::vector<BlockSequence> blocks = {{1,2,0,0,0,0,0,0,0,0}, {0, 1, 0}, {0, 0}};
    prior.setNestedState(blocks);
    BlockMove move = {0, 1, 0, -1, 1};
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}


TEST_F(TestNestedBlockUniformHyperPrior, reduceHierarchy_forSomeState_returnSameLikelihood){
    prior.sample();
    while(prior.getDepth() != 5)
        prior.sample();
    const auto nestedState = prior.getNestedState();
    double before = prior.getLogLikelihood();
    const auto reducedNestedState = prior.reduceHierarchy(nestedState);
    prior.setNestedState(reducedNestedState);
    double after = prior.getLogLikelihood();
    EXPECT_NEAR(before, after, 1e-6);
}
