#include "gtest/gtest.h"
#include <vector>
#include <iostream>

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
        EXPECT_TRUE(prior.getMaxBlockCountFromPartition(b) <= prior.getNestedBlockCountAtLevel(l));
        size_t N = prior.getNestedBlockCountAtLevel(l - 1);
        EXPECT_EQ(b.size(), N);
        ++l;
    }
}

TEST_F(TestNestedBlockPrior, getNestedStateAtLevel){
    prior.sample();
    for (size_t l=0; l<prior.getDepth(); ++l)
        EXPECT_EQ(prior.getNestedStateAtLevel(l), prior.getNestedState()[l]);
}

TEST_F(TestNestedBlockPrior, getNestedVertexCounts){
    prior.sample();
    const auto& vertexCounts = prior.getNestedVertexCounts();
    EXPECT_EQ(vertexCounts.size(), prior.getDepth());
}

TEST_F(TestNestedBlockPrior, getBlockOfIdx_forSomeVertexAtAllLevels_returnCorrectIndex){
    prior.sample();
    while(prior.getDepth()!=3)
        prior.sample();
    BaseGraph::VertexIndex vertex = 7;
    BlockIndex index = prior.getNestedStateAtLevel(0)[vertex];
    EXPECT_EQ(prior.getBlockOfIdx(vertex, 0), index);
    index = prior.getNestedStateAtLevel(1)[index];
    EXPECT_EQ(prior.getBlockOfIdx(vertex, 1), index);
    index = prior.getNestedStateAtLevel(2)[index];
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

TEST_F(TestNestedBlockPrior, applyLabelMove_forSomeBlockMoveAtFirstLevel_returnConsistentState){
    prior.sample();
    BaseGraph::VertexIndex vertex = 0;
    Level level = 0;
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
    BlockMove move(0, prevIndex, nextIndex, 0, level);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(vertex, level), nextIndex);
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forSomeBlockMoveAtSomeLevel_returnConsistentState){
    prior.sample();
    while(prior.getDepth() != 3)
        prior.sample();
    BaseGraph::VertexIndex vertex = 0;
    Level level = 1;
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
    BlockMove move(0, prevIndex, nextIndex, 0, level);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(vertex, level), nextIndex);
    EXPECT_NO_THROW(prior.checkConsistency());
}



TEST_F(TestNestedBlockPrior, applyLabelMove_forSomeBlockMoveAddingBlockAtFirstLevelNotCreatingNewLevel_returnConsistentState){
    prior.sample();
    while(prior.getDepth() < 2)
        prior.sample();
    BaseGraph::VertexIndex vertex = 0;
    Level level = 0;
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level), nextIndex = prior.getNestedBlockCountAtLevel(level);
    BlockMove move(0, prevIndex, nextIndex, 1, level);
    std::vector<size_t> prevBlockCounts = prior.getNestedBlockCount();
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(vertex, level), nextIndex);
    for (Level l=0; l<prior.getDepth(); ++l){
        size_t expected = prevBlockCounts[l] + ((l == level) ? 1 : 0) ;
        EXPECT_EQ(expected, prior.getNestedBlockCountAtLevel(l));
    }
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forSomeBlockMoveAddingBlockAtSomeLevelNotCreatingNewLevel_returnConsistentState){
    prior.sample();
    while(prior.getDepth() != 4)
        prior.sample();
    BaseGraph::VertexIndex vertex = 0;
    Level level = 1;
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = prior.getNestedBlockCountAtLevel(level);
    BlockMove move(0, prevIndex, nextIndex, 1, level);
    std::vector<size_t> prevBlockCounts = prior.getNestedBlockCount();
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(vertex, level), nextIndex);
    for (Level l=0; l<prior.getDepth(); ++l){
        if (l == level)
        EXPECT_EQ(prevBlockCounts[l] + 1, prior.getNestedBlockCountAtLevel(l));
        else
        EXPECT_EQ(prevBlockCounts[l], prior.getNestedBlockCountAtLevel(l));
    }
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, applyLabelMove_forSomeBlockMoveAddingBlockCreatingNewLevel_returnConsistentState){
    prior.sample();
    while(prior.getDepth() != 4)
        prior.sample();
    BaseGraph::VertexIndex vertex = 0;
    Level level = 3;
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = prior.getNestedBlockCountAtLevel(level);
    BlockMove move(0, prevIndex, nextIndex, 1, level);
    std::vector<size_t> prevBlockCounts = prior.getNestedBlockCount();
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(vertex, level), nextIndex);
    for (Level l=0; l<prior.getDepth(); ++l){
        if (l == level + 1)
            EXPECT_EQ(1, prior.getNestedBlockCountAtLevel(l));
        else if (l == level)
            EXPECT_EQ(prevBlockCounts[l] + 1, prior.getNestedBlockCountAtLevel(l));
        else
            EXPECT_EQ(prevBlockCounts[l], prior.getNestedBlockCountAtLevel(l));
    }
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockPrior, isValideBlockMove_forMoveAtFirstLevelNotCreatingLabel_returnCorrectValue){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 0;
    for (size_t i=0; i<NUM_SAMPLES; ++i){
        prior.sample();
        while (prior.getDepth() != 3)
            prior.sample();
        BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
        BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
        int it = 0;
        while( it < 100 and prior.getNestedStateAtLevel(level+1)[prevIndex] == prior.getNestedStateAtLevel(level+1)[nextIndex] ){
            nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
            ++it;
        }

        BlockMove move(0, prevIndex, nextIndex, 0, level);
        bool expectedIsValue = (prior.getNestedStateAtLevel(move.level + 1)[move.prevLabel] == prior.getNestedStateAtLevel(move.level + 1)[move.nextLabel]);
        EXPECT_EQ(prior.isValideBlockMove(move), expectedIsValue);
    }
}

TEST_F(TestNestedBlockPrior, isValideBlockMove_forMoveAtSomeLevelNotCreatingLabel_returnCorrectValue){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 1;
    for (size_t i=0; i<NUM_SAMPLES; ++i){
        prior.sample();
        while (prior.getDepth() != 4)
            prior.sample();
        BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
        BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
        int it = 0;
        while( it < 100 and prior.getNestedStateAtLevel(level+1)[prevIndex] == prior.getNestedStateAtLevel(level+1)[nextIndex] ){
            nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
            ++it;
        }

        BlockMove move(0, prevIndex, nextIndex, 0, level);
        bool expectedIsValue = (prior.getNestedStateAtLevel(move.level + 1)[move.prevLabel] == prior.getNestedStateAtLevel(move.level + 1)[move.nextLabel]);
        EXPECT_EQ(prior.isValideBlockMove(move), expectedIsValue);
    }
}

class TestNestedBlockUniformPrior: public ::testing::Test {
    public:

        size_t GRAPH_SIZE=10, NUM_SAMPLES=100;
        NestedBlockUniformPrior prior = NestedBlockUniformPrior(GRAPH_SIZE);

        bool expectConsistencyError = false;
        void SetUp() {
            prior.checkSafety();
            prior.sample();
        }
        void TearDown(){
            if (not expectConsistencyError)
                prior.checkConsistency();
        }
};

TEST_F(TestNestedBlockUniformPrior, setSize_forNewSize_returnStateWithNewSize) {
    prior.setSize(11);
    prior.sample();
    EXPECT_EQ(prior.getState().size(), 11);
}

TEST_F(TestNestedBlockUniformPrior, getLogLikelihood_returnSomeOfLogLikelihoodAtEachLevel) {
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = 0;
    for (Level l=0; l<prior.getDepth(); ++l)
        expectedLogLikelihood += prior.getLogLikelihoodAtLevel(l);
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
}

TEST_F(TestNestedBlockUniformPrior, getLogLikelihoodAtLevel_forAllLevel_returnCorrectValue) {
    for (Level l=0; l<prior.getDepth(); ++l){
        double actualLogLikelihood = prior.getLogLikelihoodAtLevel(l);
        double graphSize = prior.getNestedBlockCountAtLevel(l - 1);
        double blockCount = prior.getNestedBlockCountAtLevel(l);
        double expectedLogLikelihood = -graphSize * log(blockCount);
        EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
    }
}

TEST_F(TestNestedBlockUniformPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveAtFirstLevel_returnCorrectRatio) {
    BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
    Level level = 0;
    BlockMove move = {0, prior.getBlockOfIdx(0, level), sampleUniformly<BlockIndex>(0, prior.getNestedBlockCountAtLevel(level) - 1), 0, level};
    while( not prior.isValideBlockMove(move) ){
         move = {0, prior.getBlockOfIdx(0, level), sampleUniformly<BlockIndex>(0, prior.getNestedBlockCountAtLevel(level) - 1), 0, level};
     }
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}




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
        double graphSize = prior.getNestedBlockCountAtLevel(l - 1);
        double blockCount = prior.getNestedBlockCountAtLevel(l);
        double expectedLogLikelihood = -logBinomialCoefficient(graphSize - 1, blockCount - 1) - logFactorial(graphSize);

        for (const auto& nr : prior.getNestedVertexCountsAtLevel(l)){
            expectedLogLikelihood += logFactorial(nr.second);
        }

        EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
    }
}

TEST_F(TestNestedBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forValidLabelMoveAtFirstLevel_returnCorrectRatio) {
    BaseGraph::VertexIndex vertex = 0; //sampleUniformly(0, prior.getSize());
    Level level = 0;
    BlockMove move = {0, prior.getBlockOfIdx(0, level), sampleUniformly<BlockIndex>(0, prior.getNestedBlockCountAtLevel(level) - 1), 0, level};
    while( not prior.isValideBlockMove(move) ){
         move = {0, prior.getBlockOfIdx(0, level), sampleUniformly<BlockIndex>(0, prior.getNestedBlockCountAtLevel(level) - 1), 0, level};
     }
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter-logLikelihoodBefore, 1e-6);
}
