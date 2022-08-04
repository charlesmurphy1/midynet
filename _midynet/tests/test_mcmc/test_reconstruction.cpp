#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/dynamics/sis.hpp"
#include "FastMIDyNet/proposer/edge/hinge_flip.h"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/proposer/nested_label/uniform.hpp"
#include "FastMIDyNet/mcmc/reconstruction.hpp"
#include "FastMIDyNet/rng.h"
#include "../fixtures.hpp"

using namespace std;

namespace FastMIDyNet{

class TestGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    DummyErdosRenyiGraph randomGraph = DummyErdosRenyiGraph();
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    DummySISDynamics dynamics = DummySISDynamics(randomGraph);
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
    DummySBMGraph graphPrior = DummySBMGraph();
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    GibbsUniformLabelProposer<BlockIndex> blockProposer = GibbsUniformLabelProposer<BlockIndex>();
    DummyLabeledSISDynamics dynamics = DummyLabeledSISDynamics(graphPrior);
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


class TestNestedVertexLabeledGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    DummyNestedSBMGraph graphPrior = DummyNestedSBMGraph();
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    RestrictedUniformNestedBlockProposer blockProposer = RestrictedUniformNestedBlockProposer();
    DummyNestedSISDynamics dynamics = DummyNestedSISDynamics(graphPrior);
    NestedVertexLabeledGraphReconstructionMCMC<BlockIndex> mcmc = NestedVertexLabeledGraphReconstructionMCMC<BlockIndex>(dynamics, edgeProposer, blockProposer);
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

// TEST_F(TestNestedVertexLabeledGraphReconstructionMCMC, doMetropolisHastingsStep){
//     mcmc.doMetropolisHastingsStep();
// }
//
// TEST_F(TestNestedVertexLabeledGraphReconstructionMCMC, doMHSweep){
//     mcmc.doMHSweep(10);
// }


} // FastMIDyNet
