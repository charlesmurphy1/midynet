#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_EDGES = 50;
static const int NUM_VERTICES = 50;

class TestErdosRenyiModelBase: public::testing::Test{
    public:
        EdgeCountDeltaPrior edgeCountPrior = {NUM_EDGES};
        ErdosRenyiModelBase randomGraph = ErdosRenyiModelBase(NUM_VERTICES, edgeCountPrior);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_F(TestErdosRenyiModelBase, sample_getGraphWithCorrectNumberOfEdges){
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getGraph().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}


TEST_F(TestErdosRenyiModelBase, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectLogLikelihoodRatio){
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

TEST_F(TestErdosRenyiModelBase, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectLogLikelihoodRatio){
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

TEST_F(TestErdosRenyiModelBase, isCompatible_forGraphSampledFromSBM_returnTrue){
    randomGraph.sample();
    auto g = randomGraph.getGraph();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(TestErdosRenyiModelBase, isCompatible_forEmptyGraph_returnFalse){
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(TestErdosRenyiModelBase, isCompatible_forGraphWithOneEdgeMissing_returnFalse){
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
