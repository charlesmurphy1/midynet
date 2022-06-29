#include "gtest/gtest.h"

#include "fixtures.hpp"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/mcmc/community.hpp"
#include "FastMIDyNet/mcmc/callbacks/action.h"
#include "FastMIDyNet/rng.h"

using namespace std;

namespace FastMIDyNet{


class TestVertexLabelMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    DummySBM randomGraph = DummySBM();
    GibbsUniformLabelProposer<BlockIndex> proposer = GibbsUniformLabelProposer<BlockIndex>();
    VertexLabelMCMC<BlockIndex> mcmc = VertexLabelMCMC<BlockIndex>(randomGraph, proposer);
    CheckConsistencyOnSweep callback;
    bool expectConsistencyError = false;
    void SetUp(){
        seed(1);
        randomGraph.sample();
        mcmc.insertCallBack("check_consistency", callback);
        mcmc.setUp();
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
        mcmc.tearDown();
    }
};

TEST_F(TestVertexLabelMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestVertexLabelMCMC, doMHSweep){
    mcmc.doMHSweep(1000);
}

}
