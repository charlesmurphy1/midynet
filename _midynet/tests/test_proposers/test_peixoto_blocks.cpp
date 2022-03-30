#include "gtest/gtest.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

const double NEW_BLOCK_PROBABILITY = .1;
const double SHIFT = 1.;
const size_t BLOCK_COUNT = 3;
const size_t GRAPH_SIZE = 100, EDGE_COUNT=100;

class TestBlockPeixotoProposer: public::testing::Test {
    public:
        BlockCountPoissonPrior blockCountPrior = {BLOCK_COUNT};
        BlockUniformPrior blockPrior = {GRAPH_SIZE, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {EDGE_COUNT};
        EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
        StochasticBlockModelFamily randomGraph = {GRAPH_SIZE, blockPrior, edgeMatrixPrior};

        BlockPeixotoProposer blockProposer = FastMIDyNet::BlockPeixotoProposer(NEW_BLOCK_PROBABILITY, SHIFT);
        size_t numSamples = 1000;

        void SetUp() {
            seedWithTime();
            randomGraph.sample();
            while (randomGraph.getBlockCount() ==1) randomGraph.sample();

            blockProposer.setUp(randomGraph);
        }

        FastMIDyNet::BlockIndex findBlockMove(BaseGraph::VertexIndex idx){
            FastMIDyNet::BlockIndex blockIdx = randomGraph.getBlockOfIdx(idx);
            if (blockIdx == randomGraph.getBlockCount() - 1) return blockIdx - 1;
            else return blockIdx + 1;
        }
};

TEST_F(TestBlockPeixotoProposer, proposeMove_ForVertexIndex0_returnBlockMove) {
    for (size_t i = 0; i < numSamples; i++) {
        FastMIDyNet::BlockMove move = blockProposer.proposeMove(0);
        EXPECT_EQ(move.vertexIdx, 0);
        EXPECT_EQ(move.prevBlockIdx, randomGraph.getBlockOfIdx(0));
    }
}

TEST_F(TestBlockPeixotoProposer, getLogProposalProbRatio_forAllBlockMoveOfIdx0_ProposalsAreNormalized) {
    double sum = 0;
    for (size_t s=0 ; s <= randomGraph.getBlockCount() ; ++s){
        FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), s};
        if (s == randomGraph.getBlockCount()) ++move.addedBlocks;
        double logProposal = blockProposer.getLogProposalProb(move);
        sum += exp(logProposal);
    }

    EXPECT_FLOAT_EQ(sum, 1);
}

TEST_F(TestBlockPeixotoProposer, getReverseLogProposalProbRatio_fromSomeBlockMove_returnCorrectRatio) {
    for (size_t i = 0; i < numSamples; i++) {
        FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), findBlockMove(0), 0};
        FastMIDyNet::BlockMove reverseMove = {0, move.nextBlockIdx, move.prevBlockIdx, 0};
        double actual = blockProposer.getReverseLogProposalProb(move);
        randomGraph.applyBlockMove(move);
        double expected = blockProposer.getLogProposalProb(reverseMove);
        EXPECT_FLOAT_EQ(actual, expected);
    }
}

TEST_F(TestBlockPeixotoProposer, getReverseLogProposalProbRatio_fromSomeBlockMoveCreatingNewBlock_returnCorrectRatio) {
    for (size_t i = 0; i < numSamples; i++) {
        FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), randomGraph.getBlockCount(), 1};
        FastMIDyNet::BlockMove reverseMove = {0, move.nextBlockIdx, move.prevBlockIdx, -1};
        double actual = blockProposer.getReverseLogProposalProb(move);
        randomGraph.applyBlockMove(move);
        double expected = blockProposer.getLogProposalProb(reverseMove);
        EXPECT_FLOAT_EQ(actual, expected);
    }
}

}
