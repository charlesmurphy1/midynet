#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_BLOCKS = 5;
static const int NUM_EDGES = 100;
static const int NUM_VERTICES = 50;

class TestDegreeCorrectedStochasticBlockModelFamily: public::testing::Test{
    public:
        BlockCountPoissonPrior blockCountPrior = {NUM_BLOCKS};
        BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
        DegreeUniformPrior degreePrior = {blockPrior, edgeMatrixPrior};

        BaseGraph::VertexIndex vertexIdx = 4;

        BaseGraph::Edge findEdge(){
            const auto& graph = randomGraph.getState();
            BaseGraph::Edge edge;
            BaseGraph::VertexIndex neighborIdx;
            for ( auto idx: graph ){
                if (graph.getDegreeOfIdx(idx) > 0){
                    auto neighbor = *graph.getNeighboursOfIdx(idx).begin();
                    neighborIdx = neighbor.vertexIndex;
                    edge = {idx, neighborIdx};
                    return edge;
                }
            }
            throw std::invalid_argument("State of randomGraph has no edge.");
        }

        FastMIDyNet::BlockIndex findBlockMove(BaseGraph::VertexIndex idx){
            FastMIDyNet::BlockIndex blockIdx = randomGraph.getBlockOfIdx(idx);
            if (blockIdx == randomGraph.getBlockCount() - 1) return blockIdx - 1;
            else return blockIdx + 1;
        }

        DegreeCorrectedStochasticBlockModelFamily randomGraph = DegreeCorrectedStochasticBlockModelFamily(NUM_VERTICES);
        void SetUp() {
            randomGraph.setBlockPrior(blockPrior);
            randomGraph.setEdgeMatrixPrior(edgeMatrixPrior);
            randomGraph.setDegreePrior(degreePrior);
            randomGraph.sample();
        }
};


TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, sampleState_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sampleState();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, sample_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore;
    if ( randomGraph.getState().isEdgeIdx(addedEdge) ) addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    else addedEdgeMultBefore = 0;

    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forAddedSelfLoop){
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forRemovedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forRemovedEdgeAndAddedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    BaseGraph::Edge addedEdge = {20, 21};
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyMove(move);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forIdentityBlockMove_doNothing){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 0};
    randomGraph.applyMove(move);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
    if (prevBlockIdx == randomGraph.getBlockCount() - 1) nextBlockIdx --;
    else nextBlockIdx ++;

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 0};
    randomGraph.applyMove(move);
    EXPECT_NE(randomGraph.getBlocks()[vertexIdx], prevBlockIdx);
    EXPECT_EQ(randomGraph.getBlocks()[vertexIdx], nextBlockIdx);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getBlockCount();
    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 1};
    randomGraph.applyMove(move);
    EXPECT_NE(randomGraph.getBlocks()[vertexIdx], prevBlockIdx);
    EXPECT_EQ(randomGraph.getBlocks()[vertexIdx], nextBlockIdx);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlockCount();
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockMove move = {vertexIdx, nextBlockIdx, prevBlockIdx, 1};
    randomGraph.applyMove(move); // creating block before destroying it
    move = {vertexIdx, prevBlockIdx, nextBlockIdx, -1};
    randomGraph.applyMove(move);
    EXPECT_EQ(randomGraph.getBlocks()[vertexIdx], nextBlockIdx);
    EXPECT_NE(randomGraph.getBlocks()[vertexIdx], prevBlockIdx);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio){
    randomGraph.applyMove({{}, {{0, 0}}});
    FastMIDyNet::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    randomGraph.applyMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio){
    randomGraph.applyMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {{0,0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forIdentityBlockMove_return0){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 0};
    EXPECT_NEAR(randomGraph.getLogLikelihoodRatio(move), 0, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
    if (prevBlockIdx == randomGraph.getBlockCount() - 1) nextBlockIdx --;
    else nextBlockIdx ++;
    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 0};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getBlockCount();

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 1};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getBlocks()[vertexIdx];
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getBlockCount();

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx, 1};
    randomGraph.applyMove(move);
    move = {vertexIdx, nextBlockIdx, prevBlockIdx, -1};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}
