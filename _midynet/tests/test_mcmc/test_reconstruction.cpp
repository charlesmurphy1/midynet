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
    DummyGraphPrior randomGraph = DummyGraphPrior();
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    DummyDynamics dynamics = DummyDynamics(randomGraph);
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    bool expectConsistencyError = false;
    void SetUp(){
        seed(1);
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
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    BlockUniformProposer blockProposer = BlockUniformProposer();
    DummyLabeledDynamics dynamics = DummyLabeledDynamics(randomGraph);
    VertexLabeledGraphReconstructionMCMC<BlockIndex> mcmc = VertexLabeledGraphReconstructionMCMC<BlockIndex>(dynamics, edgeProposer, blockProposer);
    bool expectConsistencyError = false;
    void SetUp(){
        seedWithTime();
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
