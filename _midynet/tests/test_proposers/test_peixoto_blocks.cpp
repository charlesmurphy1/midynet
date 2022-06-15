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
const size_t BLOCK_COUNT = 2;
const size_t GRAPH_SIZE = 100, EDGE_COUNT=100;

class DummyBlockPeixotoProposer: public BlockPeixotoProposer{
public:
    using BlockPeixotoProposer::BlockPeixotoProposer;
    IntMap<std::pair<BlockIndex, BlockIndex>> getEdgeMatrixDiff(const BlockMove& move) const {
        return BlockPeixotoProposer::getEdgeMatrixDiff(move);
    }
    IntMap<BlockIndex> getEdgeCountsDiff(const BlockMove& move) const {
        return BlockPeixotoProposer::getEdgeCountsDiff(move);
    }
};

class TestBlockPeixotoProposer: public::testing::Test {
    public:
        BlockCountDeltaPrior blockCountPrior = {BLOCK_COUNT};
        BlockUniformPrior blockPrior = {GRAPH_SIZE, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {EDGE_COUNT};
        EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
        StochasticBlockModelFamily randomGraph = {GRAPH_SIZE, blockPrior, edgeMatrixPrior};

        DummyBlockPeixotoProposer proposer = FastMIDyNet::DummyBlockPeixotoProposer(NEW_BLOCK_PROBABILITY, SHIFT);
        size_t numSamples = 10;

        void SetUp() {
            seedWithTime();
            randomGraph.sample();
            while (randomGraph.getBlockCount() != BLOCK_COUNT) randomGraph.sample();

            proposer.setUp(randomGraph);
            proposer.checkSafety();
        }
        void TearDown() {
            proposer.checkConsistency();
        }

        FastMIDyNet::BlockIndex findBlockMove(BaseGraph::VertexIndex idx){
            FastMIDyNet::BlockIndex blockIdx = randomGraph.getBlockOfIdx(idx);
            if (blockIdx == randomGraph.getBlockCount() - 1) return blockIdx - 1;
            else return blockIdx + 1;
        }
};

// TEST_F(TestBlockPeixotoProposer, proposeMove_ForVertexIndex0_returnBlockMove) {
//     for (size_t i = 0; i < numSamples; i++) {
//         FastMIDyNet::BlockMove move = proposer.proposeMove(0);
//         EXPECT_EQ(move.vertexIdx, 0);
//         EXPECT_EQ(move.prevBlockIdx, randomGraph.getBlockOfIdx(0));
//     }
// }

// TEST_F(TestBlockPeixotoProposer, getLogProposalProbRatio_forAllBlockMoveOfIdx0_ProposalsAreNormalized) {
//     double sum = 0;
//     for (size_t s=0 ; s <= randomGraph.getBlockCount() ; ++s){
//         FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), s};
//         if (s == randomGraph.getBlockCount()) ++move.addedBlocks;
//         double logProposal = proposer.getLogProposalProb(move);
//         std::cout << exp(logProposal) << ", " << std::endl;
//         sum += exp(logProposal);
//     }
//
//     EXPECT_FLOAT_EQ(sum, 1);
// }

TEST_F(TestBlockPeixotoProposer, getEdgeMatrixDiff_returnCorrectDiff){
    for (size_t nextBlockIdx = 0; nextBlockIdx < randomGraph.getBlockCount(); ++nextBlockIdx){
        FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), nextBlockIdx, 0};
        auto edgeMatrixDiff = proposer.getEdgeMatrixDiff(move);
        auto actualEdgeMatrix = randomGraph.getEdgeMatrix();
        for (auto diff : edgeMatrixDiff)
            actualEdgeMatrix[diff.first.first][diff.first.second] += diff.second;
        randomGraph.applyBlockMove(move);
        proposer.applyBlockMove(move);
        auto expectedEdgeMatrix = randomGraph.getEdgeMatrix();
        EXPECT_EQ(actualEdgeMatrix, expectedEdgeMatrix);
    }
}

TEST_F(TestBlockPeixotoProposer, getReverseLogProposalProb_AllBlockMove_returnCorrectProb) {
    for (size_t nextBlockIdx = 0; nextBlockIdx < randomGraph.getBlockCount(); ++nextBlockIdx){
        FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), nextBlockIdx, 0};
        FastMIDyNet::BlockMove reverseMove = {0, move.nextBlockIdx, move.prevBlockIdx, 0};
        double actual = proposer.getReverseLogProposalProb(move);
        randomGraph.applyBlockMove(move);
        proposer.applyBlockMove(move);
        double expected = proposer.getLogProposalProb(reverseMove);
        randomGraph.applyBlockMove(reverseMove);
        EXPECT_FLOAT_EQ(actual, expected);
    }
}

TEST_F(TestBlockPeixotoProposer, getReverseLogProposalProbRatio_fromSomeBlockMoveCreatingNewBlock_returnCorrectRatio) {
    for (size_t i = 0; i < numSamples; i++) {
        FastMIDyNet::BlockMove move = {0, randomGraph.getBlockOfIdx(0), i};
        FastMIDyNet::BlockMove reverseMove = {0, move.nextBlockIdx, move.prevBlockIdx};
        double actual = proposer.getReverseLogProposalProb(move);

        move.display();
        std::cout << "B_before=" << randomGraph.getBlockCount() << std::endl;
        std::cout << "nr_before=" << randomGraph.getVertexCountsInBlocks().display() << std::endl;

        randomGraph.applyBlockMove(move);
        proposer.applyBlockMove(move);
        std::cout << "B_after=" << randomGraph.getBlockCount() << std::endl;
        std::cout << "nr_after=" << randomGraph.getVertexCountsInBlocks().display() << std::endl;
        double expected = proposer.getLogProposalProb(reverseMove);
        EXPECT_FLOAT_EQ(actual, expected);
        if (abs(actual - expected) > 1e-3)
            break;
    }
}

}
