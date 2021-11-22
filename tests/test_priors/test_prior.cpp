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
        double getLogLikelihood() const { return m_state; }
        double getLogPrior() { return 0; }
        void checkSelfConsistency() const {}
        const bool getIsProcessed() const { return m_isProcessed; }
        void computationFinished() { m_isProcessed = false; }
};

class TestPrior: public ::testing::Test {
    public:
        DummyPrior prior;
        void SetUp() { prior.setState(3); }
};

TEST_F(TestPrior, IsProcessed_afterLogJointComputation_returnIsProcessedIsTrue){
    prior.getLogJoint();
    EXPECT_TRUE(prior.getIsProcessed());
}
