#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "fixtures.hpp"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{

class TestRandomGraphMCMC: public::testing::Test{
public:
    size_t N=10, B=3, E=20;
    BlockCountPoissonPrior blockCountPrior = {(double)B};
    BlockUniformHyperPrior blockPrior = {N, blockCountPrior};
    EdgeCountDeltaPrior edgeCountPrior = {E};
    EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(N, blockPrior, edgeMatrixPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    BlockUniformProposer blockProposer = BlockUniformProposer();
    RandomGraphMCMC mcmc = RandomGraphMCMC();
    bool expectConsistencyError = false;
    void SetUp(){
        mcmc.setRandomGraph(randomGraph);
        mcmc.setBlockProposer(blockProposer);
        mcmc.setEdgeProposer(edgeProposer);
        randomGraph.sample();
        mcmc.setUp();
        mcmc.checkSafety();

    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
        mcmc.tearDown();
    }
};

TEST_F(TestRandomGraphMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestRandomGraphMCMC, doMHSweep){
    std::cout << "block count before: " << randomGraph.getBlockCount() << std::endl;
    displayVector(randomGraph.getBlocks(), "b_before (score=" + std::to_string(randomGraph.getLogJoint()) + ")");

    // auto out = mcmc.doMHSweep(100);
    size_t success = 0, failure = 0;
    for (size_t i=0; i<100; ++i){
        if (mcmc.doMetropolisHastingsStep()) ++success;
        else ++failure;
    }
    std::cout << "block count after: " << randomGraph.getBlockCount() << std::endl;
    displayVector(randomGraph.getBlocks(), "b_after (score=" + std::to_string(randomGraph.getLogJoint()) + ")");
    std::cout << "success: " << success << ", failure: " << failure << std::endl;
}


} // FastMIDyNet
