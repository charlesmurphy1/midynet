#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;


class ErdosRenyiModelTest: public::testing::Test{
    public:
        const size_t NUM_VERTICES = 50, NUM_EDGES = 50;
        ErdosRenyiModel randomGraph = ErdosRenyiModel(NUM_VERTICES, NUM_EDGES);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_F(ErdosRenyiModelTest, sample_getGraphWithCorrectNumberOfEdges){
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getState().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}


TEST_F(ErdosRenyiModelTest, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectLogLikelihoodRatio){
    auto graph = randomGraph.getState();

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

TEST_F(ErdosRenyiModelTest, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectLogLikelihoodRatio){
    auto graph = randomGraph.getState();

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

TEST_F(ErdosRenyiModelTest, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(ErdosRenyiModelTest, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(ErdosRenyiModelTest, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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

TEST_F(ErdosRenyiModelTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError){
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));

}
