#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_BLOCKS = 3;
static const int NUM_EDGES = 100;
static const int NUM_VERTICES = 50;

class TestStochasticBlockModelFamily: public::testing::Test{
    public:
        BlockCountPoissonPrior blockCountPrior = {NUM_BLOCKS};
        BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        LabelGraphErdosRenyiPrior labelGraphPrior = {edgeCountPrior, blockPrior};

        BaseGraph::Edge findEdge(){
            const auto& graph = randomGraph.getGraph();
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

        BaseGraph::VertexIndex vertex = 4;

        StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(NUM_VERTICES);
        void SetUp() {
            randomGraph.setLabelGraphPrior(labelGraphPrior);
            randomGraph.sample();
            while (randomGraph.getVertexCounts().size() == 1) randomGraph.sample();

        }
};


TEST_F(TestStochasticBlockModelFamily, sampleState_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getGraph();
        randomGraph.sample();
        auto nextGraph = randomGraph.getGraph();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestStochasticBlockModelFamily, sample_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getGraph();
        randomGraph.sample();
        auto nextGraph = randomGraph.getGraph();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore = randomGraph.getGraph().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getGraph().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forAddedSelfLoop){
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getGraph().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getGraph().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forRemovedEdge){
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getGraph().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getGraph().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forRemovedEdgeAndAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getGraph().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultBefore = randomGraph.getGraph().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getGraph().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultAfter = randomGraph.getGraph().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);

}

TEST_F(TestStochasticBlockModelFamily, applyMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forIdentityBlockMove_doNothing){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = prevLabel;

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = findBlockMove(vertex);

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();
    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getVertexCounts().size();
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockMove move = {vertex, nextLabel, prevLabel};
    randomGraph.applyLabelMove(move); // creating block before destroying it
    move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabelOfIdx(vertex), nextLabel);
    EXPECT_NE(randomGraph.getLabelOfIdx(vertex), prevLabel);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    FastMIDyNet::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio){
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    FastMIDyNet::GraphMove move = {{{0, 2}}, {{0,0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forIdentityBlockMove_return0){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = prevLabel;

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};

    EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = findBlockMove(vertex);
    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio){

    FastMIDyNet::BlockIndex prevLabel = randomGraph.getLabelOfIdx(vertex);
    FastMIDyNet::BlockIndex nextLabel = randomGraph.getVertexCounts().size();

    FastMIDyNet::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    move = {vertex, nextLabel, prevLabel};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getGraph();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestStochasticBlockModelFamily, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(TestStochasticBlockModelFamily, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
    randomGraph.sample();
    auto g = randomGraph.getGraph();
    for (auto vertex: g){
        for (auto neighbor: g.getNeighboursOfIdx(vertex)){
            g.removeEdgeIdx(vertex, neighbor.vertexIndex);
            break;
        }
    }
    EXPECT_FALSE(randomGraph.isCompatible(g));
}


TEST_F(TestStochasticBlockModelFamily, setLabels_forSomeRandomLabels_returnConsistentState){
    size_t N = randomGraph.getSize();
    size_t B = randomGraph.getLabelCount();
    std::vector<BlockIndex> newLabels(N);
    std::uniform_int_distribution<BlockIndex> dist(0, B-1);
    for (size_t v=0; v<N; ++v)
        newLabels[v] = dist(rng);
    randomGraph.setLabels(newLabels);
    EXPECT_EQ(randomGraph.getLabels(), newLabels);
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}
