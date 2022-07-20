#include "gtest/gtest.h"

#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/mcmc/community.hpp"
#include "FastMIDyNet/mcmc/callbacks/action.h"
#include "FastMIDyNet/rng.h"
#include "../fixtures.hpp"

using namespace std;

namespace FastMIDyNet{


class TestVertexLabelMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    DummySBMGraph randomGraph = DummySBMGraph();
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

TEST_F(TestVertexLabelMCMC, setLabels_noThrow){
    size_t N = randomGraph.getSize();
    size_t B = randomGraph.getLabelCount();
    std::vector<BlockIndex> newLabels(N);
    std::uniform_int_distribution<BlockIndex> dist(0, B-1);
    for (size_t v=0; v<N; ++v)
        newLabels[v] = dist(rng);
    mcmc.setLabels(newLabels);
    EXPECT_EQ(mcmc.getLabels(), newLabels);
    EXPECT_NO_THROW(mcmc.checkConsistency());
}

}
