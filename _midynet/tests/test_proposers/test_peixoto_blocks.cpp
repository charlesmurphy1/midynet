#include "gtest/gtest.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/label/peixoto.hpp"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

// const double NEW_BLOCK_PROBABILITY = .1;
// const double SHIFT = 1.;
// const size_t BLOCK_COUNT = 3;
// const size_t GRAPH_SIZE = 100, EDGE_COUNT=100;

// class DummyBlockPeixotoProposer: public BlockPeixotoProposer{
// public:
//     using BlockPeixotoProposer::BlockPeixotoProposer;
//     IntMap<std::pair<BlockIndex, BlockIndex>> getEdgeMatrixDiff(const BlockMove& move) const {
//         return BlockPeixotoProposer::getEdgeMatrixDiff(move);
//     }
//     IntMap<BlockIndex> getEdgeCountsDiff(const BlockMove& move) const {
//         return BlockPeixotoProposer::getEdgeCountsDiff(move);
//     }
// };
//
// class TestBlockPeixotoProposer: public::testing::Test {
//     public:
//         BlockCountDeltaPrior blockCountPrior = {BLOCK_COUNT};
//         BlockUniformPrior blockPrior = {GRAPH_SIZE, blockCountPrior};
//         EdgeCountPoissonPrior edgeCountPrior = {EDGE_COUNT};
//         EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
//         StochasticBlockModelFamily randomGraph = {GRAPH_SIZE, blockPrior, edgeMatrixPrior};
//
//         DummyBlockPeixotoProposer proposer = DummyBlockPeixotoProposer(NEW_BLOCK_PROBABILITY, SHIFT);
//         size_t numSamples = 10;
//
//         void SetUp() {
//             // seedWithTime();
//             randomGraph.sample();
//             while (blockPrior.getBlockCount() != BLOCK_COUNT) randomGraph.sample();
//
//             proposer.setUp(randomGraph);
//             proposer.checkSafety();
//         }
//         void TearDown() {
//             proposer.checkConsistency();
//         }
//
//         BlockIndex findLabelMove(BaseGraph::VertexIndex idx){
//             BlockIndex blockIdx = randomGraph.getLabelOfIdx(idx);
//             if (blockIdx == blockPrior.getBlockCount() - 1) return blockIdx - 1;
//             else return blockIdx + 1;
//         }
// };
//
// // TEST_F(TestBlockPeixotoProposer, proposeMove_ForVertexIndex0_returnLabelMove) {
// //     for (size_t i = 0; i < numSamples; i++) {
// //         BlockMove move = proposer.proposeMove(0);
// //         EXPECT_EQ(move.vertexIdx, 0);
// //         EXPECT_EQ(move.prevLabel, randomGraph.getLabelOfIdx(0));
// //     }
// // }
//
// TEST_F(TestBlockPeixotoProposer, getEdgeMatrixDiff_returnCorrectDiff){
//     for (size_t nextLabel = 0; nextLabel < blockPrior.getBlockCount(); ++nextLabel){
//         BlockMove move = {0, randomGraph.getLabelOfIdx(0), nextLabel};
//         auto edgeMatrixDiff = proposer.getEdgeMatrixDiff(move);
//         auto actualEdgeMatrix = randomGraph.getLabelGraph();
//         for (auto diff : edgeMatrixDiff)
//             actualEdgeMatrix.addMultiedgeIdx(diff.first.first, diff.first.second, diff.second);
//         randomGraph.applyLabelMove(move);
//         proposer.applyLabelMove(move);
//         auto expectedEdgeMatrix = randomGraph.getLabelGraph();
//         EXPECT_EQ(actualEdgeMatrix.getAdjacencyMatrix(), expectedEdgeMatrix.getAdjacencyMatrix());
//     }
// }
//
// TEST_F(TestBlockPeixotoProposer, getLogProposalProbAndSampler_returnCorrectProb){
//     CounterMap<size_t> counter;
//     for (int i=0; i<10000; ++i)
//         counter.increment(proposer.proposeMove(0).nextLabel);
//     for (auto k: counter){
//         BlockMove m = {0, randomGraph.getLabelOfIdx(0), k.first};
//         double prob = exp(proposer.getLogProposalProb(m));
//         EXPECT_NEAR((double) k.second / counter.getSum(), prob, 0.05);
//     }
// }
//
// TEST_F(TestBlockPeixotoProposer, getReverseLogProposalProb_AllLabelMove_returnCorrectProb) {
//     for (size_t nextLabel = 0; nextLabel < blockPrior.getBlockCount(); ++nextLabel){
//         BlockMove move = {0, randomGraph.getLabelOfIdx(0), nextLabel};
//         BlockMove reverseMove = {0, move.nextLabel, move.prevLabel};
//         double actual = proposer.getReverseLogProposalProb(move);
//         randomGraph.applyLabelMove(move);
//         proposer.applyLabelMove(move);
//         double expected = proposer.getLogProposalProb(reverseMove);
//         randomGraph.applyLabelMove(reverseMove);
//         proposer.applyLabelMove(reverseMove);
//         EXPECT_FLOAT_EQ(actual, expected);
//     }
// }
//
// TEST_F(TestBlockPeixotoProposer, getReverseLogProposalProbRatio_fromSomeLabelMoveCreatingNewBlock_returnCorrectRatio) {
//     for (size_t i = 0; i < numSamples; i++) {
//         BlockMove move = {0, randomGraph.getLabelOfIdx(0), i};
//         BlockMove reverseMove = {0, move.nextLabel, move.prevLabel};
//         double actual = proposer.getReverseLogProposalProb(move);
//
//         randomGraph.applyLabelMove(move);
//         proposer.applyLabelMove(move);
//         double expected = proposer.getLogProposalProb(reverseMove);
//
//         randomGraph.applyLabelMove(reverseMove);
//         proposer.applyLabelMove(reverseMove);
//
//         EXPECT_FLOAT_EQ(actual, expected);
//         if (abs(actual - expected) > 1e-3)
//             break;
//     }
// }

}
