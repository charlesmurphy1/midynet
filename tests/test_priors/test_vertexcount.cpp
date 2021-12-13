#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/rng.h"


const size_t SIZE=25;
const double POISSON_MEAN=5;

using namespace std;
using namespace FastMIDyNet;

class DummyVertexCountPrior: public VertexCountPrior {
    public:
        using VertexCountPrior::VertexCountPrior;
        void samplePriors() { m_blockCountPrior.sample(); }
        double getLogLikelihood() const { return 0.; }

        void sampleState() {
            vector<size_t> vertexCount(getBlockCount(),1);
            vertexCount[0] = getSize() - getBlockCount() + 1;
            setState(vertexCount);
        }
        double getLogLikelihoodFromState(const size_t& state) const { return state; }
        double getLogPrior() { return 0; }

        void checkSelfConsistency() const {}
        bool getIsProcessed() { return m_isProcessed; }
        double getLogLikelihoodRatioFromBlockMove(const BlockMove& ) { return 0.; }
        void _createBlock(){ createBlock(); }
        void _destroyBlock(const BlockIndex& idx){ destroyBlock(idx); }
};

class TestVertexCountPrior: public ::testing::Test {
    public:
        BlockCountPoissonPrior blockCountPrior = {POISSON_MEAN};
        DummyVertexCountPrior prior = {SIZE, blockCountPrior};
        void SetUp() {
            setSeedWithTime();
            prior.sample();
            prior.computationFinished();

            while (prior.getBlockCount() == 1){
                prior.sample();
                prior.computationFinished();
            }
        }
};

TEST_F(TestVertexCountPrior, applyBlockMoveOnState_forSomeBlockMove_stateChangesAccordingly) {
    BlockMove move = {0, 0, 1, 0};
    prior.applyBlockMoveToState(move);
    EXPECT_EQ(prior.getState()[0], SIZE - prior.getBlockCount());
    EXPECT_EQ(prior.getState()[1], 2);
    for(size_t i = 2; i < prior.getBlockCount(); ++i){
        EXPECT_EQ(prior.getState()[i], 1);
    }
}

TEST_F(TestVertexCountPrior, applyBlockMoveOnState_forSomeBlockMoveCreatingBlock_stateChangesAccordingly) {
    auto B = prior.getBlockCount();
    BlockMove move = {0, 0, B, 1};
    prior.applyBlockMoveToState(move);
    EXPECT_EQ(prior.getState()[0], SIZE - B);
    EXPECT_EQ(prior.getState()[B], 1);
    for(size_t i = 2; i < B; ++i){
        EXPECT_EQ(prior.getState()[i], 1);
    }
}

TEST_F(TestVertexCountPrior, createBlock) {
    prior._createBlock();
    EXPECT_EQ(prior.getState().size(), prior.getBlockCount() + 1);
    EXPECT_EQ(prior.getState()[prior.getBlockCount()], 0);
}

TEST_F(TestVertexCountPrior, destroyBlock) {
    prior._destroyBlock(0);
    EXPECT_EQ(prior.getState().size(), prior.getBlockCount() - 1);
    size_t sum = 0;
    for (auto nr : prior.getState()) { sum += nr; }
    EXPECT_EQ(sum, prior.getBlockCount() - 1);
}


class TestVertexCountUniformPrior: public ::testing::Test {
    public:
        BlockCountPoissonPrior blockCountPrior = {POISSON_MEAN};
        VertexCountUniformPrior prior = {SIZE, blockCountPrior};
        void SetUp() {
            prior.sample();
            prior.computationFinished();

            while (prior.getBlockCount() == 1){
                prior.sample();
                prior.computationFinished();
            }
        }
};

TEST_F(TestVertexCountUniformPrior, getLogLikelihood){
    EXPECT_LE(prior.getLogLikelihood(), 0);
}

TEST_F(TestVertexCountUniformPrior, sampleState){
    prior.sampleState();
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestVertexCountUniformPrior, getLogLikelihood_returnCorrectLogLikehood){
    size_t n = SIZE, b = prior.getBlockCount();
    EXPECT_LE(prior.getLogLikelihood(), 0);
    EXPECT_EQ(prior.getLogLikelihood(), -logBinomialCoefficient(n - 1, b - 1));
}

TEST_F(TestVertexCountUniformPrior, getLogLikelihoodRatioFromBlockMove_forSomeBlockMove_returnCorrectLogLikehoodRatio){
    BlockMove move = {0, 0, 1, 0};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromBlockMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyBlockMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_EQ(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore);
}

TEST_F(TestVertexCountUniformPrior, checkSelfConsistency_forChangeInSize_throwConsistencyError){
    vector<size_t> newNr = prior.getState();
    newNr.pop_back();
    prior.setState(newNr);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
}
