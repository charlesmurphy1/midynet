#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
// #include "FastMIDyNet/utility.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS;


class DummyPrior: public FastMIDyNet::Prior<size_t> {
    public:
        void samplePriors() { }
        void sampleState() { }
        double getLogLikelihood() const override { return m_state; }
        double getLogPrior() const override{ return 0; }
        void checkSelfConsistency() const override {}
        const bool getIsProcessed() const { return m_isProcessed; }
        void computationFinished() const override{ m_isProcessed = false; }
        void checkSafety() const override{ }
};

class TestPrior: public ::testing::Test {
    public:
        DummyPrior prior;
        void SetUp() { prior.setState(3); }
};

TEST_F(TestPrior, IsProcessedWhileNotRoot_afterLogJointComputation_returnIsProcessedIsTrue){
    prior.isRoot(false);
    prior.getLogJoint();
    EXPECT_TRUE(prior.getIsProcessed());
}

TEST_F(TestPrior, IsProcessedWhileRoot_afterLogJointComputation_returnIsProcessedIsFalse){
    prior.isRoot(true);
    prior.getLogJoint();
    EXPECT_FALSE(prior.getIsProcessed());
}
