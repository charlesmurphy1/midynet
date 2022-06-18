#include "gtest/gtest.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/label_proposer/uniform.hpp"
#include "FastMIDyNet/types.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

const double NEW_BLOCK_PROBABILITY = .1;
const size_t BLOCK_COUNT = 2;
const size_t GRAPH_SIZE = 5, EDGE_COUNT=6;
const FastMIDyNet::BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0};

class TestBlockUniformProposer: public::testing::Test {
    public:
        BlockCountPoissonPrior blockCountPrior = {BLOCK_COUNT};
        BlockUniformPrior blockPrior = {GRAPH_SIZE, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {EDGE_COUNT};
        EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
        StochasticBlockModelFamily randomGraph = {GRAPH_SIZE, blockPrior, edgeMatrixPrior};

        BlockUniformProposer proposer = FastMIDyNet::BlockUniformProposer(NEW_BLOCK_PROBABILITY);

        void SetUp() {
            blockCountPrior.setState(BLOCK_COUNT);
            blockPrior.setState(BLOCK_SEQUENCE);
            proposer.setUp(randomGraph);
            proposer.checkSafety();
        }
        void TearDown() {
            proposer.checkConsistency();
        }
};

TEST_F(TestBlockUniformProposer, getLogProposalProbRatio_sameBlockmove_return0) {
    FastMIDyNet::BlockMove move = {0, 0, 0};
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestBlockUniformProposer, getLogProposalProbRatio_moveBetweenExistingAndNonEmptyBlocks_return0) {
    FastMIDyNet::BlockMove move = {0, 0, 1};
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestBlockUniformProposer, getLogProposalProbRatio_createNewBlock_returnCorrectRatio) {
    FastMIDyNet::BlockMove move = {0, 0, 2};
    EXPECT_EQ(proposer.getLogProposalProbRatio(move),
            -log(NEW_BLOCK_PROBABILITY)+log(1-NEW_BLOCK_PROBABILITY)-log(BLOCK_COUNT));
}

TEST_F(TestBlockUniformProposer, getLogProposalProbRatio_destroyBlock_returnCorrectRatio) {
    FastMIDyNet::BlockMove move = {2, 1, 0};
    EXPECT_EQ(proposer.getLogProposalProbRatio(move),
            log(BLOCK_COUNT-1)-log(1-NEW_BLOCK_PROBABILITY)+log(NEW_BLOCK_PROBABILITY));
}

}
