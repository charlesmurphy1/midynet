#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/rng.h"


namespace FastMIDyNet{


class DummyMCMC: public MCMC{
private:
    std::uniform_real_distribution<double> m_uniform = std::uniform_real_distribution<double>(0, 1);
public:
    void doMetropolisHastingsStep(){
        m_lastLogJointRatio = 0;
        m_lastLogAcceptance = -log(2);
        if (m_uniform(rng) < exp(m_lastLogAcceptance))
            m_lastIsAccepted = true;
        else
            m_lastIsAccepted = false;

    }
    double getLogLikelihood() { return 1; }
    double getLogPrior() { return 2; }
    double getLogJoint() { return getLogLikelihood() + getLogPrior(); }
    void sample() { }
};

class TestMCMC: public::testing::Test{
public:
    DummyMCMC mcmc = DummyMCMC();
    void setUp(){
        mcmc.setUp();
        setSeed(time(NULL));
    }
    void tearDown(){
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
