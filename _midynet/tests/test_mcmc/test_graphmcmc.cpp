#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "fixtures.hpp"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{

class TestStochasticBlockGraphMCMC: public::testing::Test{
public:

    DummyRandomGraph randomGraph = DummyRandomGraph(10);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    UniformBlockProposer blockProposer = UniformBlockProposer();
    RandomGraphMCMC mcmc = RandomGraphMCMC(randomGraph, edgeProposer, blockProposer);
    void SetUp(){
        seed(time(NULL));
        mcmc.sample();
        mcmc.setUp();

    }
    void TearDown(){
        mcmc.tearDown();
    }
};

TEST_F(TestStochasticBlockGraphMCMC, doMetropolisHastingsStep){
    auto blocksBefore = mcmc.getBlocks();
    while ( not mcmc.isLastAccepted() || mcmc.getLastLogJointRatio() == 0 )
        mcmc.doMetropolisHastingsStep();
    auto blocksAfter = mcmc.getBlocks();

    size_t numDiff = 0;
    for (size_t i=0; i < blocksBefore.size(); ++i){
        if (blocksBefore[i] != blocksAfter[i]) ++numDiff;
    }
    EXPECT_EQ(numDiff, 1);
}


} // FastMIDyNet
