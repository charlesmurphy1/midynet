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

    BlockCountDeltaPrior blockCountPrior = {3};
    BlockUniformHyperPrior blockPrior = {10, blockCountPrior};
    EdgeCountDeltaPrior edgeCountPrior = {10};
    EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(10, blockPrior, edgeMatrixPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    BlockUniformProposer blockProposer = BlockUniformProposer();
    RandomGraphMCMC mcmc = RandomGraphMCMC(randomGraph, edgeProposer, blockProposer);
    bool expectConsistencyError = false;
    void SetUp(){
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


} // FastMIDyNet
