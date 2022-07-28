#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "../fixtures.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

using namespace FastMIDyNet;

class TestNestedLabelGraphPrior: public ::testing::Test {
    public:

        size_t EDGE_COUNT=10, GRAPH_SIZE=10;
        size_t NUM_SAMPLES=100;
        EdgeCountDeltaPrior edgeCountPrior = EdgeCountDeltaPrior(EDGE_COUNT);
        NestedStochasticBlockLabelGraphPrior prior = NestedStochasticBlockLabelGraphPrior(GRAPH_SIZE, edgeCountPrior);
        MultiGraph graph;

        bool expectConsistencyError = false;
        void SetUp() {
            prior.checkSafety();
            graph = generateMultiGraphErdosRenyi(GRAPH_SIZE, EDGE_COUNT);
            prior.sample();
            prior.setGraph(graph);
        }
        void TearDown(){ }

        BlockMove proposeNestedBlockMove(
            NestedLabelGraphPrior& prior,
            BaseGraph::VertexIndex id,
            Level level,
            size_t depth=4,
            bool creatingNewBlock=false,
            bool destroyingBlock=false
        ){

            size_t it = 0;
            while (it < 100){
                graph = generateMultiGraphErdosRenyi(GRAPH_SIZE, EDGE_COUNT);
                prior.sample();
                prior.setGraph(graph);
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
                } else{
                    s = sampleUniformly(0, (int) prior.getNestedBlockCount(level) - 1);
                    if (prior.getNestedVertexCounts(level)[r] == 1 and not destroyingBlock)
                        continue;
                    else if (prior.getNestedVertexCounts(level)[r] != 1 and destroyingBlock)
                        continue;
                    else if (prior.getNestedVertexCounts(level)[r] == 1 and destroyingBlock)
                        addedLabels = -1;
                }
                BlockMove move = {id, r, s, addedLabels, level};
                if (prior.getNestedBlockPrior().isValidBlockMove(move) and r != s)
                    return move;
                ++it;
            }
            throw std::logic_error("Could not create valid move.");
        }
};

TEST_F(TestNestedLabelGraphPrior, sampleState_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihood_returnSumOfLogLikelihoodAtEaclLevel){
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = 0;
    for (Level l=0; l<prior.getDepth(); ++l)
        expectedLogLikelihood += prior.getLogLikelihoodAtLevel(l);
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihood_forLastLevel_returnCorrectValue){
    while(prior.getDepth() == 1)
        prior.sample();
    Level level = prior.getDepth() - 1;
    double actualLogLikelihood = prior.getLogLikelihoodAtLevel(level);
    size_t N = prior.getNestedState(level - 1).getSize();
    double expectedLogLikelihood = -logMultisetCoefficient(N * (N + 1) / 2, 2 * prior.getEdgeCount());
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihood_forAnyLevelOtherThanLast_returnCorrectValue){
    for (Level level=0; level < prior.getDepth() - 1; ++level){
        double actualLogLikelihood = prior.getLogLikelihoodAtLevel(level);
        double expectedLogLikelihood = 0;
        size_t err, ers;
        BlockIndex r, s;
        for (const auto& nr : prior.getNestedBlockPrior().getNestedVertexCounts(level)) {
            r = nr.first;
            err = prior.getNestedState(level).getEdgeMultiplicityIdx(r, r);
            expectedLogLikelihood -= logMultisetCoefficient(nr.second * (nr.second + 1) / 2, 2 * err);
            for (const auto& ns : prior.getNestedBlockPrior().getNestedVertexCounts(level)) {
                s = ns.first;
                if (r >= s)
                    continue;
                ers = prior.getNestedState(level).getEdgeMultiplicityIdx(r, s);
                expectedLogLikelihood -= logMultisetCoefficient(nr.second * ns.second, ers);
            }
        }
        EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
    }
}

TEST_F(TestNestedLabelGraphPrior, setGraph_noThrow){
    EXPECT_EQ(prior.getGraph(), graph);
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
}


TEST_F(TestNestedLabelGraphPrior, applyGraphMove_forAddedEdge_noThrow){
    GraphMove move = {{}, {{0, 1}}};
    BlockIndex r = prior.getNestedBlocks(0)[0], s = prior.getNestedBlocks(0)[1];
    auto stateBefore = prior.getNestedState(0);
    prior.applyGraphMove(move);
    auto stateAfter = prior.getNestedState(0);
    EXPECT_EQ(stateBefore.getEdgeMultiplicityIdx(r, s), stateAfter.getEdgeMultiplicityIdx(r, s) - 1);
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
}

TEST_F(TestNestedLabelGraphPrior, applyGraphMove_forAddedSelfLoop_noThrow){
    GraphMove move = {{}, {{0, 0}}};
    BlockIndex r = prior.getNestedBlocks(0)[0];
    auto stateBefore = prior.getNestedState(0);
    prior.applyGraphMove(move);
    auto stateAfter = prior.getNestedState(0);
    EXPECT_EQ(stateBefore.getEdgeMultiplicityIdx(r, r), stateAfter.getEdgeMultiplicityIdx(r, r) - 1);
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
}

TEST_F(TestNestedLabelGraphPrior, applyLabelMove_forIdentityMoveAtAnyLevel_doNothing){
    for(Level level=0; level < prior.getDepth(); ++level){
        BlockMove move = {0, prior.getBlockOfIdx(0, level), prior.getBlockOfIdx(0, level), 0, level};
        prior.applyLabelMove(move);
        EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
    }
}

TEST_F(TestNestedLabelGraphPrior, applyLabelMove_forMoveNotChangingBlockCountAtAnyLevel_noThrow){

    size_t depth = 4;
    for(Level level=0; level < 1; ++level){
        BlockMove move = proposeNestedBlockMove(prior, 0, level, depth);
        prior.applyLabelMove(move);
        EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
        EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
    }

}

TEST_F(TestNestedLabelGraphPrior, applyLabelMove_forMoveChangingBlockCountAtAnyLevel_noThrow){
    size_t depth = 4;
    for(Level level=0; level < depth - 1; ++level){
        BlockMove move = proposeNestedBlockMove(prior, 0, level, depth, true);
        prior.applyLabelMove(move);
        EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
        EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
    }
}

TEST_F(TestNestedLabelGraphPrior, applyLabelMove_forMoveChangingBlockCountAtAnyLevel_noThrow2){
    size_t depth = 4;
    BlockMove move = proposeNestedBlockMove(prior, 0, 2, depth, true);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getBlockOfIdx(move.vertexIndex, move.level), move.nextLabel);
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
}

TEST_F(TestNestedLabelGraphPrior, applyLabelMove_forMoveIncreasingDepth_noThrow){
    size_t depth = 3;
    size_t id = 0;
    Level level = depth - 1;
    BlockMove move = proposeNestedBlockMove(prior, id, level, depth, true);

    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getDepth(), depth + 1);
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectValue){
    GraphMove move = {{}, {{0, 1}}};
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihoodRatioFromGraphMove_forAddedSelfLoop_returnCorrectValue){
    GraphMove move = {{}, {{0, 0}}};
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihoodRatioFromLabelMove_forIdentityMoveAtAnyLevel_returnCorrectValue){
    size_t depth = 4;
    for (Level l=0; l<depth; ++l){
        while(prior.getDepth() != depth)
            prior.sample();
        BlockMove move = {0, prior.getBlockOfIdx(0, l), prior.getBlockOfIdx(0, l), 0, l};
        double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
        EXPECT_NEAR(expectedLogLikelihoodRatio, 0, 1e-6);
    }
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihoodRatioFromLabelMove_forMoveNotChangingBlockCountAtAnyLevel_returnCorrectValue){
    size_t depth = 4;
    for (Level l=0; l<depth - 1; ++l){
        BlockMove move = proposeNestedBlockMove(prior, 0, l, depth);
        double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
        double logLikelihoodBefore = prior.getLogLikelihood();
        prior.applyLabelMove(move);
        double logLikelihoodAfter = prior.getLogLikelihood();
        EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
    }
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihoodRatioFromLabelMove_forMoveChangingBlockCountAtAnyLevel_returnCorrectValue){
    size_t depth = 4;
    for (Level l=0; l<1; ++l){
        BlockMove move = proposeNestedBlockMove(prior, 0, l, depth, true);
        double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
        double logLikelihoodBefore = prior.getLogLikelihood();
        prior.applyLabelMove(move);
        double logLikelihoodAfter = prior.getLogLikelihood();
        EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
    }
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihoodRatioFromLabelMove_forMoveIncreasingDepth_returnCorrectValue){
    size_t depth = 4;
    BlockMove move = proposeNestedBlockMove(prior, 0, depth-1, depth, true);
    double expectedLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(expectedLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}
