#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "fixtures.hpp"
#include "FastMIDyNet/mcmc/graph_mcmc.h"

#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/dynamics/sis.h"
#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/rng.h"

using namespace std;

namespace FastMIDyNet{
size_t NUM_STEPS=10;

class TestDynamicsMCMC: public::testing::Test{
public:
    BlockCountDeltaPrior blockCountPrior = {3};
    BlockUniformHyperPrior blockPrior = {10, blockCountPrior};
    EdgeCountDeltaPrior edgeCountPrior = {10};
    EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(10, blockPrior, edgeMatrixPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    BlockUniformProposer blockProposer = BlockUniformProposer();
    RandomGraphMCMC graphmcmc = RandomGraphMCMC(randomGraph, edgeProposer, blockProposer);
    SISDynamics dynamics = SISDynamics(randomGraph, NUM_STEPS, 0.5);
    DynamicsMCMC mcmc = DynamicsMCMC(dynamics, graphmcmc, 1., 1., 0.);
    bool expectConsistencyError = false;
    void SetUp(){
        seed(time(NULL));
        dynamics.sample();
        mcmc.setUp();
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
        mcmc.tearDown();
    }
};

TEST_F(TestDynamicsMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}


} // FastMIDyNet
