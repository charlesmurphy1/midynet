#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_EDGES = 50;
static const int NUM_VERTICES = 50;

class TestErdosRenyiFamily: public::testing::Test{
    public:
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        ErdosRenyiFamily randomGraph = ErdosRenyiFamily(NUM_VERTICES, edgeCountPrior);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_F(TestErdosRenyiFamily, randomGraph_hasCorrectBlockSequence){
    auto blocks = randomGraph.getVertexLabels();
    for (auto b : blocks) EXPECT_EQ(b, 0);
}

TEST_F(TestErdosRenyiFamily, sample_getGraphWithCorrectNumberOfEdges){
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getGraph().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_F(TestErdosRenyiFamily, getLogPriorRatioFromLabelMove_returnMinusInfinity){
    BlockMove move = {0, 0, 1};
    double dS = randomGraph.getLogPriorRatioFromLabelMove(move);
    EXPECT_EQ(dS, -INFINITY);
}

TEST_F(TestErdosRenyiFamily, applyBlockMove_throwConsistencyError){
    #if DEBUG
    BlockMove move = {0, 0, 1};
    EXPECT_THROW(randomGraph.applyBlockMove(move), ConsistencyError);
    #endif
}

TEST_F(TestErdosRenyiFamily, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getGraph();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestErdosRenyiFamily, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(TestErdosRenyiFamily, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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


class TestSimpleErdosRenyiFamily: public::testing::Test{
    public:
        EdgeCountDeltaPrior edgeCountPrior = {NUM_EDGES};
        SimpleErdosRenyiFamily randomGraph = SimpleErdosRenyiFamily(NUM_VERTICES, edgeCountPrior);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_F(TestSimpleErdosRenyiFamily, sample_getGraphWithCorrectNumberOfEdges){
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getGraph().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}


TEST_F(TestSimpleErdosRenyiFamily, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectLogLikelihoodRatio){
    auto graph = randomGraph.getGraph();

    GraphMove move = {};
    for (auto vertex: graph){
        if (graph.getEdgeMultiplicityIdx(0, vertex) == 0) {
            move.addedEdges.push_back({0, vertex});
            break;
        }
    }
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);


}

TEST_F(TestSimpleErdosRenyiFamily, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    auto graph = randomGraph.getGraph();

    GraphMove move = {};
    for (auto neighbor: graph.getNeighboursOfIdx(0)){
        move.removedEdges.push_back({0, neighbor.vertexIndex});
        break;
    }

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);

}

TEST_F(TestSimpleErdosRenyiFamily, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getGraph();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestSimpleErdosRenyiFamily, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(TestSimpleErdosRenyiFamily, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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
