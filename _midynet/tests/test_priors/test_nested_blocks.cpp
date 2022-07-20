#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

using namespace FastMIDyNet;

class TestNestedBlockUniformPrior: public ::testing::Test {
    public:

        size_t GRAPH_SIZE=10;
        NestedBlockUniformPrior prior = NestedBlockUniformPrior(GRAPH_SIZE);
        void SetUp() {
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestNestedBlockUniformPrior, sampleState_returnConsistentState){
    prior.sample();
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockUniformPrior, getNestedState){
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

TEST_F(TestNestedBlockUniformPrior, getNestedStateAtLevel){
    prior.sample();
    for (size_t l=0; l<prior.getDepth(); ++l)
        EXPECT_EQ(prior.getNestedStateAtLevel(l), prior.getNestedState()[l]);
}

TEST_F(TestNestedBlockUniformPrior, getNestedVertexCounts){
    prior.sample();
    const auto& vertexCounts = prior.getNestedVertexCounts();
    EXPECT_EQ(vertexCounts.size(), prior.getDepth());
}

TEST_F(TestNestedBlockUniformPrior, creatingNewLevel_forMoveNotCreatingLevel_returnTrue){
    prior.sample();
    BlockMove move = {0, 0, 0, 1, ((int)prior.getDepth())-1};
    EXPECT_TRUE(prior.creatingNewLevel(move));
}

TEST_F(TestNestedBlockUniformPrior, creatingNewLevel_forMoveCreatingLevel_returnFalse){
    prior.sample();
    BlockMove move = {0, 0, 0, 1, 0};
    EXPECT_FALSE(prior.creatingNewLevel(move));
}

class TestNestedBlockUniformHyperPrior: public ::testing::Test {
    public:

        size_t GRAPH_SIZE=10;
        NestedBlockUniformHyperPrior prior = NestedBlockUniformHyperPrior(GRAPH_SIZE);
        void SetUp() {
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestNestedBlockUniformHyperPrior, sampleState_returnConsistentState){
    prior.sample();
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockUniformHyperPrior, getNestedState){
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

TEST_F(TestNestedBlockUniformHyperPrior, getNestedStateAtLevel){
    prior.sample();
    for (size_t l=0; l<prior.getDepth(); ++l)
        EXPECT_EQ(prior.getNestedStateAtLevel(l), prior.getNestedState()[l]);
}

TEST_F(TestNestedBlockUniformHyperPrior, getNestedVertexCounts){
    prior.sample();
    const auto& vertexCounts = prior.getNestedVertexCounts();
    EXPECT_EQ(vertexCounts.size(), prior.getDepth());
}

TEST_F(TestNestedBlockUniformHyperPrior, creatingNewLevel_forMoveNotCreatingLevel_returnTrue){
    prior.sample();
    BlockMove move = {0, 0, 0, 1, ((int)prior.getDepth())-1};
    EXPECT_TRUE(prior.creatingNewLevel(move));
}

TEST_F(TestNestedBlockUniformHyperPrior, creatingNewLevel_forMoveCreatingLevel_returnFalse){
    prior.sample();
    while(prior.getBlockCount() == 1)
        prior.sample();
    BlockMove move = {0, 0, 0, 1, 0};
    EXPECT_FALSE(prior.creatingNewLevel(move));
}
