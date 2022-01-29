#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/rng.h"


namespace FastMIDyNet{


class DummyMCMC: public MCMC{
private:
    MultiGraph graph = MultiGraph(0);
    BlockSequence blocks = BlockSequence();
    std::uniform_real_distribution<double> m_uniform = std::uniform_real_distribution<double>(0, 1);
public:
    void doMetropolisHastingsStep(){
        m_lastLogJointRatio = 0;
        m_lastLogAcceptance = -log(2);
        if (m_uniform(rng) < exp(m_lastLogAcceptance))
            m_isLastAccepted = true;
        else
            m_isLastAccepted = false;

    }
    const double getLogLikelihood() const { return 1; }
    const double getLogPrior() const { return 2; }
    const double getLogJoint() const { return getLogLikelihood() + getLogPrior(); }
    void sample() { m_hasState = true; }
    const MultiGraph& getGraph() const override { return graph; }
    const BlockSequence& getBlocks() const override { return blocks; }
};

class TestMCMC: public::testing::Test{
public:
    DummyMCMC mcmc = DummyMCMC();
    void SetUp(){
        mcmc.sample();
        mcmc.setUp();
        seed(time(NULL));
    }
    void TearDown(){
        mcmc.tearDown();
    }

};

TEST_F(TestMCMC, doMHSweep_for42Burn_mcmcStateIsUpdated){
    mcmc.doMHSweep(42);
    EXPECT_EQ(mcmc.getNumSteps(), 42);
    EXPECT_EQ(mcmc.getNumSweeps(), 1);

    mcmc.doMHSweep(42);
    EXPECT_EQ(mcmc.getNumSteps(), 84);
    EXPECT_EQ(mcmc.getNumSweeps(), 2);

    EXPECT_EQ(mcmc.getLastLogAcceptance(), -log(2));
    EXPECT_EQ(mcmc.getLastLogJointRatio(), 0);
}


} // FastMIDyNet
