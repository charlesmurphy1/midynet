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
    auto blocks = randomGraph.getBlocks();
    for (auto b : blocks) EXPECT_EQ(b, 0);
}

TEST_F(TestErdosRenyiFamily, sample_getGraphWithCorrectNumberOfEdges){
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getGraph().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_F(TestErdosRenyiFamily, getLogLikelihoodRatioFromBlockMove_returnMinusInfinity){
    BlockMove move = {0, 0, 1, 1};
    double dS = randomGraph.getLogPriorRatioFromBlockMove(move);
    EXPECT_EQ(dS, -INFINITY);
}

TEST_F(TestErdosRenyiFamily, applyBlockMove_throwConsistencyError){
    BlockMove move = {0, 0, 1, 1};
    EXPECT_THROW(randomGraph.applyBlockMove(move), ConsistencyError);
}


class TestSimpleErdosRenyiFamily: public::testing::Test{
    public:
        EdgeCountDeltaPrior edgeCountPrior = {NUM_EDGES};
        SimpleErdosRenyiFamily randomGraph = SimpleErdosRenyiFamily(NUM_VERTICES, edgeCountPrior);
        void SetUp() {
            randomGraph.samplePriors();
            std::cout << "Edge count: " << randomGraph.getEdgeCount() << std::endl; 
            randomGraph.sample();
        }
};

TEST_F(TestSimpleErdosRenyiFamily, randomGraph_hasCorrectBlockSequence){
    auto blocks = randomGraph.getBlocks();
    for (auto b : blocks) EXPECT_EQ(b, 0);
}

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
    std::cout << "Before: " << randomGraph.getEdgeCount() << std::endl;
    randomGraph.applyGraphMove(move);
    std::cout << "After: " << randomGraph.getEdgeCount() << std::endl;
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
    std::cout << "Before: " << randomGraph.getEdgeCount() << std::endl;
    randomGraph.applyGraphMove(move);
    std::cout << "After: " << randomGraph.getEdgeCount() << std::endl;
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);

}
