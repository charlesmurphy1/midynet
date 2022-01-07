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
        BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
        DegreeUniformPrior degreePrior = {blockPrior, edgeMatrixPrior};
        ConfigurationModelFamily randomGraph = ConfigurationModelFamily(degreePrior);
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
    EXPECT_EQ(randomGraph.getState().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_F(TestConfigurationModelFamily, getLogLikelihoodRatioFromBlockMove_returnMinusInfinity){
    BlockMove move = {0, 0, 1, 1};
    double dS = randomGraph.getLogPriorRatio(move);
    EXPECT_EQ(dS, -INFINITY);
}

TEST_F(TestConfigurationModelFamily, applyBlockMove_throwConsistencyError){
    BlockMove move = {0, 0, 1, 1};
    EXPECT_THROW(randomGraph.applyMove(move), ConsistencyError);
}
