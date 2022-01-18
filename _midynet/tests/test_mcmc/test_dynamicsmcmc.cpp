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
    DummyRandomGraph randomGraph = DummyRandomGraph(10);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    BlockUniformProposer blockProposer = BlockUniformProposer();
    RandomGraphMCMC graphmcmc = RandomGraphMCMC(randomGraph, edgeProposer, blockProposer);
    SISDynamics dynamics = SISDynamics(randomGraph, NUM_STEPS, 0.5);
    DynamicsMCMC mcmc = DynamicsMCMC(dynamics, graphmcmc, 1., 1., 0.);
    void SetUp(){
        seed(time(NULL));
        mcmc.sample();
        mcmc.setUp();

    }
    void TearDown(){
        mcmc.tearDown();
    }
};

TEST_F(TestDynamicsMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}


} // FastMIDyNet
