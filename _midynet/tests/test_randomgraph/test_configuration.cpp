#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/degree.h"
#include "FastMIDyNet/random_graph/configuration.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class CMParametrizedTest: public::testing::TestWithParam<bool>{
    public:
        const size_t NUM_VERTICES = 50, NUM_EDGES = 100;
        ConfigurationModelFamily randomGraph = ConfigurationModelFamily(NUM_VERTICES, NUM_EDGES);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_P(CMParametrizedTest, sample_getStateWithCorrectNumberOfEdges){
    EXPECT_EQ(randomGraph.getState().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_P(CMParametrizedTest, getLogLikelihood_returnNonPositiveValue){
    EXPECT_LE(randomGraph.getLogLikelihood(), 0);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectValue){
    GraphMove move = {{}, {{0, 1}}};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forAddedSelfLoop_returnCorrectValue){
    GraphMove move = {{}, {{0, 0}}};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}



TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectValue){
    GraphMove move = {{{0, 1}}, {}};
    randomGraph.applyGraphMove({{}, {{0, 1}}});
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forRemovedSelfLoop_returnCorrectValue){
    GraphMove move = {{{0, 0}}, {}};
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(CMParametrizedTest, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(CMParametrizedTest, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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

TEST_P(CMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError){
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}


INSTANTIATE_TEST_CASE_P(
        ConfigurationModelFamilyTests,
        CMParametrizedTest,
        ::testing::Values( false, true )
    );


TEST(CMTests, instanciateConfigurationModel_forRegularSequence){
    std::vector<size_t> degrees(100, 5);

    ConfigurationModel graph(degrees);

}
