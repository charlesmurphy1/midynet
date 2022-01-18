#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/random_graph/configuration.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_EDGES = 100;
static const int NUM_VERTICES = 50;

class TestConfigurationModelFamily: public::testing::Test{
    public:
        BlockCountDeltaPrior blockCountPrior = {1};
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        DegreeUniformPrior degreePrior = {};
        ConfigurationModelFamily randomGraph = ConfigurationModelFamily(NUM_VERTICES, edgeCountPrior, degreePrior);
        void SetUp() {
            randomGraph.sample();
        }
};

TEST_F(TestConfigurationModelFamily, randomGraph_hasCorrectBlockSequence){
    auto blocks = randomGraph.getBlocks();
    for (auto b : blocks) EXPECT_EQ(b, 0);
}

TEST_F(TestConfigurationModelFamily, sample_getGraphWithCorrectNumberOfEdges){
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getGraph().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_F(TestConfigurationModelFamily, getLogLikelihoodRatioFromBlockMove_returnMinusInfinity){
    BlockMove move = {0, 0, 1, 1};
    double dS = randomGraph.getLogPriorRatioFromBlockMove(move);
    EXPECT_EQ(dS, -INFINITY);
}

TEST_F(TestConfigurationModelFamily, applyBlockMove_throwConsistencyError){
    BlockMove move = {0, 0, 1, 1};
    EXPECT_THROW(randomGraph.applyBlockMove(move), ConsistencyError);
}
