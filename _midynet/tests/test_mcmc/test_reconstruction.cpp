#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "fixtures.hpp"
#include "FastMIDyNet/dynamics/sis.hpp"
#include "FastMIDyNet/proposer/edge/hinge_flip.h"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/mcmc/reconstruction.hpp"
#include "FastMIDyNet/rng.h"

using namespace std;

namespace FastMIDyNet{

class TestGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    DummyER randomGraph = DummyER();
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, numSteps, 0.5);
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
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

TEST_F(TestGraphReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestGraphReconstructionMCMC, doMHSweep){
    mcmc.doMHSweep(1000);
}


} // FastMIDyNet
