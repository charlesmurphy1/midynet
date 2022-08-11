#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/random_graph/hsbm.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class HSBMParametrizedTest: public::testing::TestWithParam<bool>{
public:
    const size_t NUM_VERTICES = 50, NUM_EDGES = 100;
    const bool canonical = false, stubLabeled = false;
    NestedStochasticBlockModelFamily randomGraph = NestedStochasticBlockModelFamily(
        NUM_VERTICES, NUM_EDGES, canonical, GetParam()
    );

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

    BaseGraph::VertexIndex vertex = 4;

    void SetUp() {
        while (randomGraph.getLabelCount() == 1 or randomGraph.getLabelCount() > 30)
            randomGraph.sample();
    }

    BlockMove proposeNestedBlockMove(
        BaseGraph::VertexIndex id,
        Level level,
        size_t depth=4,
        bool creatingNewBlock=false,
        bool destroyingBlock=false
    ){

        size_t it = 0;
        while (it < 100){
            randomGraph.sample();
            if (randomGraph.getDepth() != depth)
                continue;
            BlockIndex r, s;
            int addedLabels = 0;
            r = randomGraph.getLabelOfIdx(id, level);
            if (randomGraph.getNestedVertexCounts(level)[r] == 1 and not destroyingBlock)
                continue;
            if (creatingNewBlock){
                addedLabels = 1;
                s = randomGraph.getNestedLabelCount(level);
            } else{
                s = sampleUniformly(0, (int) randomGraph.getNestedLabelCount(level) - 1);
                if (randomGraph.getNestedVertexCounts(level)[r] == 1 and not destroyingBlock)
                    continue;
                else if (randomGraph.getNestedVertexCounts(level)[r] != 1 and destroyingBlock)
                    continue;
                else if (randomGraph.getNestedVertexCounts(level)[r] == 1 and destroyingBlock)
                    addedLabels = -1;
            }
            BlockMove move = {id, r, s, addedLabels, level};
            if (randomGraph.isValidLabelMove(move) and r != s)
                return move;
            ++it;
        }
        throw std::logic_error("Could not create valid move.");
    }
};


TEST_P(HSBMParametrizedTest, sampleState_graphChanges){
    for (size_t i = 0; i < 2; i++) {
        auto prevGraph = randomGraph.getState();

        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
        EXPECT_NO_THROW(randomGraph.checkConsistency());
    }
}

TEST_P(HSBMParametrizedTest, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_P(HSBMParametrizedTest, applyGraphMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST_P(HSBMParametrizedTest, applyGraphMove_forAddedSelfLoop){
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(HSBMParametrizedTest, applyGraphMove_forRemovedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_P(HSBMParametrizedTest, applyGraphMove_forRemovedEdgeAndAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);

}

TEST_P(HSBMParametrizedTest, applyGraphMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

TEST_P(HSBMParametrizedTest, applyLabelMove_forIdentityBlockMove_doNothing){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = prevLabel;

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
}

TEST_P(HSBMParametrizedTest, applyLabelMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){

    FastMIDyNet::BlockMove move = proposeNestedBlockMove(vertex, 0, 3);
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), move.prevLabel);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), move.nextLabel);
}

// TEST_P(HSBMParametrizedTest, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
//     FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
//     FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();
//     FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
//     randomGraph.applyLabelMove(move);
//     EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
// }
//
// TEST_P(HSBMParametrizedTest, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
//     FastMIDyNet::BlockIndex prevLabel = randomGraph.getVertexCounts().size();
//     FastMIDyNet::BlockIndex nextLabel = randomGraph.getLabelOfIdx(vertex);
//     FastMIDyNet::BlockMove move = {vertex, nextLabel, prevLabel};
//     randomGraph.applyLabelMove(move); // creating block before destroying it
//     move = {vertex, prevLabel, nextLabel};
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
//     EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
// }
//
TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    FastMIDyNet::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {{0,0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forIdentityBlockMove_return0){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = prevLabel;
    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};


    EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
}

TEST_P(HSBMParametrizedTest, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::BlockMove move = proposeNestedBlockMove(vertex, 0, 3);
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HSBMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(HSBMParametrizedTest, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(HSBMParametrizedTest, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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


TEST_P(HSBMParametrizedTest, setLabels_forSomeRandomLabels_returnDepletedMethodError){
    size_t N = randomGraph.getSize();
    size_t B = randomGraph.getLabelCount();
    std::vector<BlockIndex> newLabels(N);
    std::uniform_int_distribution<BlockIndex> dist(0, B-1);
    for (size_t v=0; v<N; ++v)
        newLabels[v] = dist(rng);
    EXPECT_THROW(randomGraph.setLabels(newLabels), DepletedMethodError);
}


TEST_P(HSBMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError){
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}

TEST_P(HSBMParametrizedTest, doingMetropolisHastingsWithLabels_expectNoConsistencyError){
    EXPECT_NO_THROW(doMetropolisHastingsSweepForLabels(randomGraph));
}


INSTANTIATE_TEST_CASE_P(
        NestedStochasticBlockModelFamilyTests,
        HSBMParametrizedTest,
        ::testing::Values( false, true )
    );
