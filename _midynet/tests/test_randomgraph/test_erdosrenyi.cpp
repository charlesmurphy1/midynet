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

static const int NUM_EDGES = 100;
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
    EXPECT_EQ(randomGraph.getState().getTotalEdgeNumber(), randomGraph.getEdgeCount());
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
