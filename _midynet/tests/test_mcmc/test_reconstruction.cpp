#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "fixtures.hpp"
#include "FastMIDyNet/dynamics/sis.h"
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
    SISDynamics dynamics = SISDynamics(randomGraph, numSteps, 0.5);
    GraphReconstructionMCMC mcmc = GraphReconstructionMCMC(dynamics, randomGraph, proposer);
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

class TestVertexLabeledGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    DummySBM randomGraph = DummySBM();
    SISDynamics dynamics = SISDynamics(randomGraph, numSteps, 0.5);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    BlockUniformProposer blockProposer = BlockUniformProposer();
    VertexLabeledGraphReconstructionMCMC<BlockIndex> mcmc = VertexLabeledGraphReconstructionMCMC<BlockIndex>(
        dynamics, randomGraph, edgeProposer, blockProposer
    );
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

TEST_F(TestVertexLabeledGraphReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestVertexLabeledGraphReconstructionMCMC, doMHSweep){
    mcmc.doMHSweep(1000);
}


} // FastMIDyNet
