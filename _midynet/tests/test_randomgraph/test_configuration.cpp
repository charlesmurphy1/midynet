#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/degree.h"
#include "FastMIDyNet/random_graph/configuration.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class TestConfigurationModelFamily: public::testing::Test{
    public:
        double AVG_NUM_EDGES = 100;
        size_t NUM_VERTICES = 50;
        EdgeCountPoissonPrior edgeCountPrior = {AVG_NUM_EDGES};
        DegreeUniformPrior degreePrior = {NUM_VERTICES, edgeCountPrior};
        ConfigurationModelFamily randomGraph = ConfigurationModelFamily(NUM_VERTICES, edgeCountPrior, degreePrior);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_F(TestConfigurationModelFamily, sample_getGraphWithCorrectNumberOfEdges){
    EXPECT_EQ(randomGraph.getGraph().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_F(TestConfigurationModelFamily, getLogLikelihood_returnNonPositiveValue){
    EXPECT_LE(randomGraph.getLogLikelihood(), 0);
}

TEST_F(TestConfigurationModelFamily, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectValue){
    GraphMove move = {{}, {{0, 1}}};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_F(TestConfigurationModelFamily, getLogLikelihoodRatioFromGraphMove_forAddedSelfLoop_returnCorrectValue){
    GraphMove move = {{}, {{0, 0}}};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}



TEST_F(TestConfigurationModelFamily, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectValue){
    GraphMove move = {{{0, 1}}, {}};
    randomGraph.applyGraphMove({{}, {{0, 1}}});
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_F(TestConfigurationModelFamily, getLogLikelihoodRatioFromGraphMove_forRemovedSelfLoop_returnCorrectValue){
    GraphMove move = {{{0, 0}}, {}};
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_F(TestConfigurationModelFamily, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getGraph();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestConfigurationModelFamily, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(TestConfigurationModelFamily, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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
