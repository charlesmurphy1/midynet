#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"
#include "FastMIDyNet/random_graph/prior/labeled_degree.h"
#include "FastMIDyNet/random_graph/hdcsbm.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class TestNestedDegreeCorrectedStochasticBlockModelFamily: public::testing::Test{
    public:
        const size_t NUM_VERTICES = 50, NUM_EDGES = 100;
        const bool canonical = false, useHyperPrior = true;
        NestedDegreeCorrectedStochasticBlockModelFamily randomGraph = NestedDegreeCorrectedStochasticBlockModelFamily(
            NUM_VERTICES, NUM_EDGES, useHyperPrior, canonical
        );
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
            while(randomGraph.getLabelCount() > 30)
                randomGraph.sample();
        }
};


TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, sampleState_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyGraphMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore;
    if ( randomGraph.getState().isEdgeIdx(addedEdge) ) addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    else addedEdgeMultBefore = 0;

    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyGraphMove_forAddedSelfLoop){
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyGraphMove_forRemovedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyGraphMove_forRemovedEdgeAndAddedEdge){
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

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyGraphMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyMove_forIdentityBlockMove_doNothing){
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
//
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     randomGraph.applyLabelMove(move);
// }
//
// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
//     if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1) nextBlockIdx --;
//     else nextBlockIdx ++;
//
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), prevBlockIdx);
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), nextBlockIdx);
// }
//
// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), prevBlockIdx);
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), nextBlockIdx);
// }
//
// // TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
// //     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getVertexCounts().size();
// //     FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
// //     FastMIDyNet::BlockMove move = {vertexIdx, nextBlockIdx, prevBlockIdx};
// //     randomGraph.applyLabelMove(move); // creating block before destroying it
// //     move = {vertexIdx, prevBlockIdx, nextBlockIdx};
// //     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), prevBlockIdx);
// //     randomGraph.applyLabelMove(move);
// //     EXPECT_EQ(randomGraph.getLabelOfIdx(vertexIdx), nextBlockIdx);
// // }
//
TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    FastMIDyNet::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {{0,0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forIdentityBlockMove_return0){
//
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
//
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
// }
//
// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){
//
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = prevBlockIdx;
//     if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1) nextBlockIdx --;
//     else nextBlockIdx ++;
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = randomGraph.getLogLikelihood();
//     randomGraph.applyLabelMove(move);
//     double logLikelihoodAfter = randomGraph.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
// }
//
// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio){
//
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
//
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
//
//     double logLikelihoodBefore = randomGraph.getLogLikelihood();
//     randomGraph.applyLabelMove(move);
//     double logLikelihoodAfter = randomGraph.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
// }
//
// TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio){
//
//     FastMIDyNet::BlockIndex prevBlockIdx = randomGraph.getLabelOfIdx(vertexIdx);
//     FastMIDyNet::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
//
//     FastMIDyNet::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     randomGraph.applyLabelMove(move);
//     move = {vertexIdx, nextBlockIdx, prevBlockIdx};
//     double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = randomGraph.getLogLikelihood();
//     randomGraph.applyLabelMove(move);
//     double logLikelihoodAfter = randomGraph.getLogLikelihood();
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
// }
//
//
TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestNestedDegreeCorrectedStochasticBlockModelFamily, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}
