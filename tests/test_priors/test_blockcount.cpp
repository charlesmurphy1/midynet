#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS;


class DummyBlockCountPrior: public FastMIDyNet::BlockCountPrior {
    public:
        size_t sample() { return 0; }
        double getLogLikelihood(const size_t& state) const { return state; }
        double getLogPrior() { return 0; }

        void checkSelfConsistency() const {}
        bool getIsProcessed() { return m_isProcessed; }
};

class TestBlockCountPrior: public ::testing::Test {
    public:
        DummyBlockCountPrior prior;
        void SetUp() { prior.setState(0); prior.computationFinished(); }
};

class TestBlockCountPoissonPrior: public::testing::Test{
    public:
        FastMIDyNet::BlockCountPoissonPrior prior={POISSON_MEAN};
};


TEST_F(TestBlockCountPrior, getStateAfterMove_blockMove_returnCorrectBlockNumberIfVertexInNewBlock) {
    for (auto currentBlockNumber: {1, 2, 5, 10}){
        prior.setState(currentBlockNumber);
        for ( auto nextBlockIdx : {0, 1, 2, 6}){
            std::vector<FastMIDyNet::BlockMove> blockMoves(2, FastMIDyNet::BlockMove(0, 0, nextBlockIdx));
            if (currentBlockNumber > nextBlockIdx)
                EXPECT_EQ(prior.getStateAfterMove(blockMoves), currentBlockNumber);
            else
                EXPECT_EQ(prior.getStateAfterMove(blockMoves), nextBlockIdx + 1);
        }
    }

}

TEST_F(TestBlockCountPrior, getLogLikelihoodRatio_noNewblockMove_return0) {
    prior.setState(5);
    std::vector<FastMIDyNet::BlockMove> blockMove(1, FastMIDyNet::BlockMove(0, 0, 1));
    EXPECT_EQ(prior.getLogLikelihoodRatio(blockMove), 0);
}

TEST_F(TestBlockCountPrior, getLogLikelihoodRatio_newblockMove_returnCorrectRatio) {
    prior.setState(5);
    std::vector<FastMIDyNet::BlockMove> blockMove(1, FastMIDyNet::BlockMove(0, 0, 7));
    EXPECT_EQ(prior.getLogLikelihoodRatio(blockMove), 3);
}

TEST_F(TestBlockCountPrior, applyMove_noNewblockMove_blockNumberUnchangedIsProcessedIsTrue) {
    prior.setState(5);
    std::vector<FastMIDyNet::BlockMove> blockMove(1, FastMIDyNet::BlockMove(0, 0, 2));
    prior.applyMove(blockMove);
    EXPECT_EQ(prior.getState(), 5);
}

TEST_F(TestBlockCountPrior, applyMove_newblockMove_blockNumberIncrementsIsProcessedIsTrue) {
    prior.setState(5);
    std::vector<FastMIDyNet::BlockMove> blockMove(1, FastMIDyNet::BlockMove(0, 0, 7));
    prior.applyMove(blockMove);
    EXPECT_EQ(prior.getState(), 8);
}


TEST_F(TestBlockCountPoissonPrior, getLogLikelihood_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihood(x),
                    FastMIDyNet::logPoissonPMF(x, POISSON_MEAN));
}

TEST_F(TestBlockCountPoissonPrior, getLogPrior_returns0) {
    EXPECT_DOUBLE_EQ(prior.getLogPrior(), 0);
}

TEST_F(TestBlockCountPoissonPrior, checkSelfConsistency_noError_noThrow) {
    prior.setState(0);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
    prior.setState(2);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockCountPoissonPrior, checkSelfConsistency_negativeMean_throwConsistencyError) {
    prior={-2};
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}
