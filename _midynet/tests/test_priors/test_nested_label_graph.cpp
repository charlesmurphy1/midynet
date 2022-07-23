#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

using namespace FastMIDyNet;


class TestNestedLabelGraphPrior: public ::testing::Test {
    public:

        size_t EDGE_COUNT=10, GRAPH_SIZE=10;
        size_t NUM_SAMPLES=100;
        NestedBlockUniformPrior blockPrior = NestedBlockUniformPrior(GRAPH_SIZE);
        EdgeCountDeltaPrior edgeCountPrior = EdgeCountDeltaPrior(EDGE_COUNT);
        NestedStochasticBlockLabelGraphPrior prior = NestedStochasticBlockLabelGraphPrior(edgeCountPrior, blockPrior);

        bool expectConsistencyError = false;
        void SetUp() {
            prior.checkSafety();
            prior.sample();
        }
        void TearDown(){ }
};

TEST_F(TestNestedLabelGraphPrior, sampleState_noThrow){
    prior.sample();
    EXPECT_NO_THROW(prior.checkSelfConsistencyBetweenLevels());
    // 
    // for (Level level=0; level<blockPrior.getDepth(); ++level)
    //     displayMatrix(prior.getNestedStateAtLevel(level).getAdjacencyMatrix(), "E_" + std::to_string(level), true);
}

TEST_F(TestNestedLabelGraphPrior, getLogLikelihood_returnSumOfLogLikelihoodAtEaclLevel){
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = 0;
    for (Level l=0; l<blockPrior.getDepth(); ++l)
        expectedLogLikelihood += prior.getLogLikelihoodAtLevel(l);
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
}
