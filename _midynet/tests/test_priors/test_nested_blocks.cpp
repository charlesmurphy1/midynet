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

class TestNestedBlockUniformPrior: public ::testing::Test {
    public:

        size_t GRAPH_SIZE=10;
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

TEST_F(TestNestedBlockUniformPrior, getBlockOfIdx_forSomeVertexAtAllLevels_returnCorrectIndex){
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

TEST_F(TestNestedBlockUniformPrior, creatingNewLevel_forMoveNotCreatingLevel_returnTrue){
    prior.sample();
    BlockMove move = {0, 0, 0, 1, ((int)prior.getDepth())-1};
    EXPECT_TRUE(prior.creatingNewLevel(move));
}

TEST_F(TestNestedBlockUniformPrior, creatingNewLevel_forMoveCreatingLevel_returnFalse){
    prior.sample();
    // BlockMove move = {0, 0, 0, 1, 0};
    // EXPECT_FALSE(prior.creatingNewLevel(move));
}

TEST_F(TestNestedBlockUniformPrior, applyLabelMove_forSomeBlockMoveAtFirstLevel_returnConsistentState){
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

TEST_F(TestNestedBlockUniformPrior, applyLabelMove_forSomeBlockMoveAtSomeLevel_returnConsistentState){
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



TEST_F(TestNestedBlockUniformPrior, applyLabelMove_forSomeBlockMoveAddingBlockAtFirstLevelNotCreatingNewLevel_returnConsistentState){
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

TEST_F(TestNestedBlockUniformPrior, applyLabelMove_forSomeBlockMoveAddingBlockAtSomeLevelNotCreatingNewLevel_returnConsistentState){
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

TEST_F(TestNestedBlockUniformPrior, applyLabelMove_forSomeBlockMoveAddingBlockCreatingNewLevel_returnConsistentState){
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

TEST_F(TestNestedBlockUniformPrior, isValideBlockMove_forValidMoveAtFirstLevelNotCreatingLabel_returnTrue){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 0;
    prior.sample();
    while(prior.getDepth() < 3 or prior.getNestedVertexCountsAtLevel(level + 1)[prior.getBlockOfIdx(vertex, level + 1)] == 1)
        prior.sample();
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
    int it = 0;
    while( it < 100 and prior.getNestedStateAtLevel(level+1)[prevIndex] != prior.getNestedStateAtLevel(level+1)[nextIndex] ){
        nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
        ++it;
    }
    BlockMove move(0, prevIndex, nextIndex, 0, level);
    prior.applyLabelMove(move);
    EXPECT_TRUE(prior.isValideBlockMove(move));
}

TEST_F(TestNestedBlockUniformPrior, isValideBlockMove_forValidMoveAtSomeLevelNotCreatingLabel_returnTrue){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 1;
    prior.sample();
    while(prior.getDepth() < 4 or prior.getNestedVertexCountsAtLevel(level + 1)[prior.getBlockOfIdx(vertex, level + 1)] == 1)
        prior.sample();
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
    int it = 0;
    while( it < 100 and prior.getNestedStateAtLevel(level+1)[prevIndex] != prior.getNestedStateAtLevel(level+1)[nextIndex] ){
        nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
        ++it;
    }

    BlockMove move(0, prevIndex, nextIndex, 0, level);

    prior.applyLabelMove(move);
    EXPECT_TRUE(prior.isValideBlockMove(move));
}

TEST_F(TestNestedBlockUniformPrior, isValideBlockMove_forInvalidMoveAtFirstLevelNotCreatingLabel_returnFalse){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 0;
    prior.sample();
    while(prior.getDepth() < 3 or prior.getNestedVertexCountsAtLevel(level + 1)[prior.getBlockOfIdx(vertex, level)] == 1)
        prior.sample();
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
    int it = 0;
    while( it < 100 and prior.getNestedStateAtLevel(level+1)[prevIndex] == prior.getNestedStateAtLevel(level+1)[nextIndex] ){
        nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
        ++it;
    }

    BlockMove move(0, prevIndex, nextIndex, 0, level);

    prior.applyLabelMove(move);
    EXPECT_FALSE(prior.isValideBlockMove(move));
    // expectConsistencyError = true;
}

TEST_F(TestNestedBlockUniformPrior, isValideBlockMove_forInvalidMoveAtSomeLevelNotCreatingLabel_returnFalse){
    BaseGraph::VertexIndex vertex = 0;
    Level level = 1;
    prior.sample();
    while(prior.getDepth() < 4 or prior.getNestedVertexCountsAtLevel(level + 1)[prior.getBlockOfIdx(vertex, level)] == 1)
        prior.sample();
    BlockIndex prevIndex = prior.getBlockOfIdx(vertex, level);
    BlockIndex nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
    int it = 0;
    while( it < 100 and prior.getNestedStateAtLevel(level+1)[prevIndex] == prior.getNestedStateAtLevel(level+1)[nextIndex] ){
        nextIndex = sampleUniformly((size_t) 1, prior.getNestedBlockCountAtLevel(level) - 1);
        ++it;
    }

    BlockMove move(0, prevIndex, nextIndex, 0, level);

    displayMatrix(prior.getNestedState(), "b", true);
    std::cout << move.display() << std::endl;
    prior.applyLabelMove(move);
    displayMatrix(prior.getNestedState(), "b", true);
    EXPECT_FALSE(prior.isValideBlockMove(move));
    // expectConsistencyError = true;
}

// class TestNestedBlockUniformHyperPrior: public ::testing::Test {
//     public:
//
//         size_t GRAPH_SIZE=10;
//         NestedBlockUniformHyperPrior prior = NestedBlockUniformHyperPrior(GRAPH_SIZE);
//         void SetUp() {
//             prior.checkSafety();
//         }
//         void TearDown(){
//             prior.checkConsistency();
//         }
// };
//
// TEST_F(TestNestedBlockUniformHyperPrior, sampleState_returnConsistentState){
//     prior.sample();
//     EXPECT_NO_THROW(prior.checkConsistency());
// }
//
// TEST_F(TestNestedBlockUniformHyperPrior, getNestedState){
//     prior.sample();
//     const auto& nestedBlocks = prior.getNestedState();
//     EXPECT_EQ(nestedBlocks.size(), prior.getDepth());
//     size_t l=0;
//     for (auto b: nestedBlocks){
//         EXPECT_TRUE(prior.getMaxBlockCountFromPartition(b) <= prior.getNestedBlockCountAtLevel(l));
//         size_t N = prior.getNestedBlockCountAtLevel(l - 1);
//         EXPECT_EQ(b.size(), N);
//         ++l;
//     }
// }
//
// TEST_F(TestNestedBlockUniformHyperPrior, getNestedStateAtLevel){
//     prior.sample();
//     for (size_t l=0; l<prior.getDepth(); ++l)
//         EXPECT_EQ(prior.getNestedStateAtLevel(l), prior.getNestedState()[l]);
// }
//
// TEST_F(TestNestedBlockUniformHyperPrior, getNestedVertexCounts){
//     prior.sample();
//     const auto& vertexCounts = prior.getNestedVertexCounts();
//     EXPECT_EQ(vertexCounts.size(), prior.getDepth());
// }
//
// TEST_F(TestNestedBlockUniformHyperPrior, creatingNewLevel_forMoveNotCreatingLevel_returnTrue){
//     prior.sample();
//     BlockMove move = {0, 0, 0, 1, ((int)prior.getDepth())-1};
//     EXPECT_TRUE(prior.creatingNewLevel(move));
// }
//
// TEST_F(TestNestedBlockUniformHyperPrior, creatingNewLevel_forMoveCreatingLevel_returnFalse){
//     prior.sample();
// //     while(prior.getBlockCount() == 1)
// //         prior.sample();
// //     BlockMove move = {0, 0, 0, 1, 0};
// //     EXPECT_FALSE(prior.creatingNewLevel(move));
// }
