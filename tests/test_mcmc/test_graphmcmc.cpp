#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/proposer/block_proposer/uniform_proposer.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{
size_t GRAPH_SIZE = 100;
size_t BLOCK_COUNT = 5;
size_t EDGE_COUNT = 250;

class TestStochasticBlockGraphMCMC: public::testing::Test{
public:
    UniformBlockProposer blockProposer = UniformBlockProposer(0.);
    BlockCountDeltaPrior blockCount = BlockCountDeltaPrior(BLOCK_COUNT);
    BlockUniformPrior blockPrior = BlockUniformPrior(GRAPH_SIZE, blockCount);
    EdgeCountDeltaPrior edgeCount = EdgeCountDeltaPrior(EDGE_COUNT);
    EdgeMatrixUniformPrior edgeMatrix = EdgeMatrixUniformPrior(edgeCount, blockPrior);
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(blockPrior, edgeMatrix);
    StochasticBlockGraphMCMC mcmc = StochasticBlockGraphMCMC(randomGraph, blockProposer);
    void SetUp(){
        seed(time(NULL));
        mcmc.sample();
        mcmc.setUp();

    }
    void TearDown(){
        mcmc.tearDown();
    }
};

TEST_F(TestStochasticBlockGraphMCMC, doMetropolisHastingsStep){
    auto blocksBefore = mcmc.getBlocks();
    while ( not mcmc.isLastAccepted() || mcmc.getLastLogJointRatio() == 0 )
        mcmc.doMetropolisHastingsStep();
    auto blocksAfter = mcmc.getBlocks();

    size_t numDiff = 0;
    for (size_t i=0; i < blocksBefore.size(); ++i){
        if (blocksBefore[i] != blocksAfter[i]) ++numDiff;
    }
    EXPECT_EQ(numDiff, 1);
}


} // FastMIDyNet
