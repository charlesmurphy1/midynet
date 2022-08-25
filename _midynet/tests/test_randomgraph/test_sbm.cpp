#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>
#include <cmath>

#include "../fixtures.hpp"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class SBMParametrizedTest: public::testing::TestWithParam<std::tuple<bool, bool, bool>>{
    public:
        const size_t NUM_VERTICES = 50, NUM_EDGES = 100, NUM_BLOCKS=3;
        const bool canonical = false;
        StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(
            NUM_VERTICES,
            NUM_EDGES,
            NUM_BLOCKS,
            std::get<0>(GetParam()),
            std::get<1>(GetParam()),
            canonical,
            std::get<2>(GetParam())
        );

        BaseGraph::VertexIndex vertex = 4;

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


        void SetUp() { }
};


TEST_P(SBMParametrizedTest, sampleState_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(SBMParametrizedTest, sample_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(SBMParametrizedTest, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_P(SBMParametrizedTest, applyMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forAddedSelfLoop){
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forRemovedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forRemovedEdgeAndAddedEdge){
    BaseGraph::Edge addedEdge = {10, 11};
    BaseGraph::Edge removedEdge = findEdge();
    while(addedEdge == removedEdge)
        removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);

}

TEST_P(SBMParametrizedTest, applyMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

TEST_P(SBMParametrizedTest, applyMove_forIdentityBlockMove_doNothing){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = prevLabel;

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
}

TEST_P(SBMParametrizedTest, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = findBlockMove(vertex);

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
}

TEST_P(SBMParametrizedTest, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();
    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
}

TEST_P(SBMParametrizedTest, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getVertexCounts().size();
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockMove move = {vertex, nextLabel, prevLabel};
    randomGraph.applyLabelMove(move); // creating block before destroying it
    move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    FastMIDyNet::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {{0,0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forIdentityBlockMove_return0){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = prevLabel;

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};

    EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = findBlockMove(vertex);
    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel, 1};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel, -1};
    randomGraph.applyLabelMove(move);
    move = {vertex, nextLabel, prevLabel};




    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(SBMParametrizedTest, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(SBMParametrizedTest, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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


TEST_P(SBMParametrizedTest, setLabels_forSomeRandomLabels_returnConsistentState){
    size_t N = randomGraph.getSize();
    size_t B = randomGraph.getLabelCount();
    std::vector<BlockIndex> newLabels(N);
    std::uniform_int_distribution<BlockIndex> dist(0, B-1);
    for (size_t v=0; v<N; ++v)
        newLabels[v] = dist(rng);
    randomGraph.setLabels(newLabels, false);
    EXPECT_EQ(randomGraph.getLabels(), newLabels);
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST_P(SBMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError){
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));

}

TEST_P(SBMParametrizedTest, doingMetropolisHastingsWithLabels_expectNoConsistencyError){
    EXPECT_NO_THROW(doMetropolisHastingsSweepForLabels(randomGraph));
}


INSTANTIATE_TEST_CASE_P(
        StochasticBlockModelFamilyTests,
        SBMParametrizedTest,
        ::testing::Values(
            std::make_tuple(false, false, false),
            std::make_tuple(false, true, false),
            std::make_tuple(true, false, false),
            std::make_tuple(true, true, false),
            std::make_tuple(false, false, true),
            std::make_tuple(false, true, true),
            std::make_tuple(true, false, true),
            std::make_tuple(true, true, true)
        )
    );


TEST(SBMTest, construction_returnSafeObject){
    std::vector<size_t> sizes = {10, 20, 30};
    size_t edgeCount = 100;
    double assortativity = 0.8;
    std::vector<BlockIndex> blocks = getPlantedBlocks(sizes);
    LabelGraph labelGraph = getPlantedLabelGraph(sizes.size(), edgeCount);
    StochasticBlockModel randomGraph = StochasticBlockModel(blocks, labelGraph);
    EXPECT_NO_THROW(randomGraph.checkSafety());
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST(UniformSBMTest, construction_returnSafeObject){
    seedWithTime();
    for (size_t i=0; i<1; ++i){
        StochasticBlockModelFamily randomGraph(100, 250, 0, true, true);
        EXPECT_NO_THROW(randomGraph.checkSafety());
        randomGraph.sample();
        EXPECT_NO_THROW(randomGraph.checkConsistency());
    }
}

TEST(PlantedPartitionModelTest, constructor1_noThrow){
    size_t edgeCount = 100;
    double assortativity = 0.8;
    PlantedPartitionModel randomGraph = PlantedPartitionModel({10, 20, 30}, edgeCount, assortativity);
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST(PlantedPartitionModelTest, constructor2_noThrow){
    size_t edgeCount = 100;
    double assortativity = 0.8;
    PlantedPartitionModel randomGraph = PlantedPartitionModel(60, edgeCount, 3, assortativity);
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}
