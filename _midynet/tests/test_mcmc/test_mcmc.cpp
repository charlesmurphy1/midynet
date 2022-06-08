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
    bool doMetropolisHastingsStep(){
        m_lastLogJointRatio = 0;
        m_lastLogAcceptance = -log(2);
        if (m_uniform(rng) < exp(m_lastLogAcceptance))
            m_isLastAccepted = true;
        else
            m_isLastAccepted = false;
        return m_isLastAccepted;

    }
    const double getLogLikelihood() const override { return 1; }
    const double getLogPrior() const override { return 2; }
    const double getLogJoint() const override { return getLogLikelihood() + getLogPrior(); }
    const MultiGraph& getGraph() const override { return graph; }
    const BlockSequence& getBlocks() const override { return blocks; }
};

class TestMCMC: public::testing::Test{
public:
    DummyMCMC mcmc = DummyMCMC();
    bool expectConsistencyError = false;
    void SetUp(){
        mcmc.setUp();
        seed(time(NULL));
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
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
