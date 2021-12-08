#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/proposer/blockproposer/uniform_blocks.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/rng.h"

size_t GRAPH_SIZE = 100;
size_t BLOCK_COUNT = 5;
size_t EDGE_COUNT = 250;


namespace FastMIDyNet{

class TestStochasticBlockGraphMCMC: public::testing::Test{
public:
    UniformBlockProposer blockProposer = UniformBlockProposer(GRAPH_SIZE, 0.);
    BlockCountDeltaPrior blockCount = BlockCountDeltaPrior(BLOCK_COUNT);
    BlockUniformPrior blockPrior = BlockUniformPrior(GRAPH_SIZE, blockCount);
    EdgeCountDeltaPrior edgeCount = EdgeCountDeltaPrior(EDGE_COUNT);
    EdgeMatrixUniformPrior edgeMatrix = EdgeMatrixUniformPrior(edgeCount, blockPrior);
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(blockPrior, edgeMatrix);
    StochasticBlockGraphMCMC mcmc = StochasticBlockGraphMCMC(randomGraph, blockProposer);
    void setUp(){
        setSeed(time(NULL));
        mcmc.setUp();
        mcmc.sample();

    }
    void tearDown(){
        mcmc.tearDown();
    }
};

TEST_F(TestStochasticBlockGraphMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}


} // FastMIDyNet
