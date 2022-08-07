#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/random_graph/prior/labeled_degree.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class TestDegreeCorrectedStochasticBlockModelBase: public::testing::Test{
    public:
        double AVG_NUM_BLOCKS = 5;
        double AVG_NUM_EDGES = 100;
        size_t NUM_VERTICES = 50;
        BlockCountPoissonPrior blockCountPrior = {AVG_NUM_BLOCKS};
        BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {AVG_NUM_EDGES};
        LabelGraphErdosRenyiPrior labelGraphPrior = {edgeCountPrior, blockPrior};
        VertexLabeledDegreeUniformPrior degreePrior = {labelGraphPrior};
        DegreeCorrectedStochasticBlockModelBase randomGraph = DegreeCorrectedStochasticBlockModelBase(NUM_VERTICES, degreePrior);

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
            FastMIDyNet::BlockIndex blockIdx = randomGraph.getLabelOfIdx(idx);
            if (blockIdx == randomGraph.getVertexCounts().size() - 1) return blockIdx - 1;
            else return blockIdx + 1;
        }

        void SetUp() {
            randomGraph.checkSafety();
            randomGraph.sample();
        }
};


TEST_F(TestDegreeCorrectedStochasticBlockModelBase, sampleState_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, sample_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore;
    if ( randomGraph.getState().isEdgeIdx(addedEdge) ) addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    else addedEdgeMultBefore = 0;

    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forAddedSelfLoop){
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forRemovedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forRemovedEdgeAndAddedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    BaseGraph::Edge addedEdge = {20, 21};
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forIdentityBlockMove_doNothing){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    randomGraph.applyLabelMove(move);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
    if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1) nextBlockIdx --;
    else nextBlockIdx ++;

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), prevBlockIdx);
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), nextBlockIdx);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), prevBlockIdx);
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), nextBlockIdx);
}

// TEST_F(TestDegreeCorrectedStochasticBlockModelBase, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getVertexCounts().size();
//     FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockMove move = {vertexIdx, nextBlockIdx, prevBlockIdx};
//     randomGraph.applyLabelMove(move); // creating block before destroying it
//     move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), prevBlockIdx);
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), nextBlockIdx);
// }

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    FastMIDyNet::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {{0,0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forIdentityBlockMove_return0){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
    if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1) nextBlockIdx --;
    else nextBlockIdx ++;
    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
    FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();

    FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    randomGraph.applyLabelMove(move);
    move = {vertexIdx, nextBlockIdx, prevBlockIdx};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}


TEST_F(TestDegreeCorrectedStochasticBlockModelBase, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestDegreeCorrectedStochasticBlockModelBase, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}
TEST_F(TestDegreeCorrectedStochasticBlockModelBase, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
    randomGraph.sample();
    auto g = randomGraph.getState();
    for (auto vertex: g){
        for (auto neighbor: g.getNeighboursOfIdx(vertex)){
            g.removeEdgeIdx(vertex, neighbor.vertexIndex);
            break;
        }
    }
    EXPECT_FALSE(randomGraph.isCompatible(g));
}
