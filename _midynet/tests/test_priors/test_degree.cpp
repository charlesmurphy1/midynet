#include "gtest/gtest.h"
#include <vector>
#include <iostream>


#include "fixtures.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

using namespace FastMIDyNet;

const double GRAPH_SIZE=7;
const double BLOCK_COUNT=5;
const double POISSON_MEAN=5;
const BlockSequence BLOCK_SEQ={0,0,0,0,1,1,1};
const double TOL=1E-8;

namespace FastMIDyNet{

class DummyDegreePrior: public DegreePrior {
    public:
        DummyDegreePrior(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
        DegreePrior(blockPrior, edgeMatrixPrior) {};

        void sampleState() {
            DegreeSequence degreeSeq(m_blockPriorPtr->getSize(), 0);
            degreeSeq[0] = m_edgeMatrixPriorPtr->getEdgeCount();
            degreeSeq[6] = m_edgeMatrixPriorPtr->getEdgeCount();
            setState(degreeSeq);
        }
        const double getLogLikelihood() const { return 0; }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const { return 0; }
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const { return 0; }
        void applyGraphMoveToState(const GraphMove& move) { DegreePrior::applyGraphMoveToState(move); }
        void applyGraphMoveToDegreeCounts(const GraphMove& move) { DegreePrior::applyGraphMoveToDegreeCounts(move); }
        void applyLabelMoveToDegreeCounts(const BlockMove& move) { DegreePrior::applyLabelMoveToDegreeCounts(move); }


};

class TestDegreePrior: public ::testing::Test {
    public:

        BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
        BlockUniformPrior blockPrior = BlockUniformPrior(GRAPH_SIZE, blockCountPrior);
        EdgeCountPoissonPrior edgeCountPrior = EdgeCountPoissonPrior(POISSON_MEAN);
        EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
        DummyDegreePrior prior = DummyDegreePrior(blockPrior, edgeMatrixPrior);

        bool expectConsistencyError = false;
        void SetUp() {
            prior.sample();
            prior.checkSafety();
        }
        void TearDown(){
            if (not expectConsistencyError)
                prior.checkConsistency();
        }
};

TEST_F(TestDegreePrior, setGraph_forHouseGraph_applyChangesToDegreeSequence){
    prior.setGraph(getUndirectedHouseMultiGraph());
    DegreeSequence expectedDegreeSeq = {4, 3, 5, 5, 2, 3, 0};
    DegreeSequence actualDegreeSeq = prior.getState();
    for (size_t i=0; i < 7; ++i){
        EXPECT_EQ(expectedDegreeSeq[i], actualDegreeSeq[i]);
    }
}

TEST_F(TestDegreePrior, computeDegreeCounts_forLocalDegreeSeqNBlockSeq_returnCorrectDegreeCounts){
    blockPrior.setState({0,0,0,0,1,1,1});
    // auto degreeCounts = prior.computeDegreeCounts(prior.getState(), prior.getBlockPrior().getState());
    // EXPECT_EQ(degreeCounts.size(), 2);
    //
    // EXPECT_EQ(degreeCounts[0].size(), 2);
    // EXPECT_FALSE(degreeCounts[0].isEmpty(0));
    // EXPECT_EQ(degreeCounts[0].get(0), 3);
    // EXPECT_FALSE(degreeCounts[0].isEmpty(prior.getEdgeMatrixPrior().getEdgeCount()));
    // EXPECT_EQ(degreeCounts[0].get(prior.getEdgeMatrixPrior().getEdgeCount()), 1);
    //
    // EXPECT_EQ(degreeCounts[1].size(), 2);
    // EXPECT_FALSE(degreeCounts[1].isEmpty(0));
    // EXPECT_EQ(degreeCounts[1].get(0), 2);
    // EXPECT_FALSE(degreeCounts[1].isEmpty(prior.getEdgeMatrixPrior().getEdgeCount()));
    // EXPECT_EQ(degreeCounts[1].get(prior.getEdgeMatrixPrior().getEdgeCount()), 1);
}

TEST_F(TestDegreePrior, applyGraphMoveToState_ForAddedEdge_returnCorrectDegreeSeq){
    GraphMove move = {{}, {{0,1}}};
    auto k0Before = prior.getState()[0];
    auto k1Before = prior.getState()[1];
    prior.applyGraphMoveToState(move);
    auto k0After = prior.getState()[0];
    auto k1After = prior.getState()[1];
    EXPECT_EQ(k0After, k0Before + 1);
    EXPECT_EQ(k1After, k1Before + 1);

    expectConsistencyError = true;
}

TEST_F(TestDegreePrior, applyGraphMoveToState_ForRemovedEdge_returnCorrectDegreeSeq){
    GraphMove move = {{{0,6}}, {}};
    auto k0Before = prior.getState()[0];
    auto k6Before = prior.getState()[6];
    prior.applyGraphMoveToState(move);
    auto k0After = prior.getState()[0];
    auto k6After = prior.getState()[6];
    EXPECT_EQ(k0After, k0Before - 1);
    EXPECT_EQ(k6After, k6Before - 1);
    expectConsistencyError = true;

}

TEST_F(TestDegreePrior, applyGraphMoveToState_ForRemovedEdgeNAddedEdge_returnCorrectDegreeSeq){
    GraphMove move = {{{0,6}}, {{0, 1}}};
    auto k0Before = prior.getState()[0];
    auto k1Before = prior.getState()[1];
    auto k6Before = prior.getState()[6];
    prior.applyGraphMoveToState(move);
    auto k0After = prior.getState()[0];
    auto k1After = prior.getState()[1];
    auto k6After = prior.getState()[6];
    EXPECT_EQ(k0After, k0Before);
    EXPECT_EQ(k1After, k1Before + 1);
    EXPECT_EQ(k6After, k6Before - 1);
    expectConsistencyError = true;

}

TEST_F(TestDegreePrior, applyGraphMoveToDegreeCounts_forAddedEdge_returnCorrectDegreeCounts){
    blockPrior.setState(BLOCK_SEQ);
    GraphMove move = {{}, {{0,1}}};
    size_t E = prior.getEdgeMatrixPrior().getEdgeCount();
    auto expected = prior.getDegreeCounts();
    expected.decrement({0, 0}); expected.increment({0, 1});
    expected.decrement({0, E}); expected.increment({0, E+1});
    prior.applyGraphMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCounts();

    // for (size_t r = 0; r < blockPrior.getBlockCount(); ++r){
    //     EXPECT_TRUE(expected[r] == actual[r]);
    // }
    for (auto nk : expected)
        EXPECT_EQ(nk.second, actual.get(nk.first));
    expectConsistencyError = true;

}

TEST_F(TestDegreePrior, applyGraphMoveToDegreeCounts_forRemovedEdge_returnCorrectDegreeCounts){
    while(blockPrior.getBlockCount()==1) prior.sample();
    GraphMove move = {{{0, 4}}, {}};
    size_t E = prior.getEdgeMatrixPrior().getEdgeCount();
    auto expected = prior.getDegreeCounts();
    prior.applyGraphMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCounts();

    BlockIndex r = blockPrior.getBlockOfIdx(0), s = blockPrior.getBlockOfIdx(0);
    size_t ki = prior.getState()[0],  kj = prior.getState()[0];

    EXPECT_EQ(expected.get({r, ki}) - 1, actual.get({r, ki}));
    EXPECT_EQ(expected.get({r, ki - 1}) + 1, actual.get({r, ki - 1}));

    EXPECT_EQ(expected.get({s, kj}) - 1, actual.get({s, kj}));
    EXPECT_EQ(expected.get({s, kj - 1}) + 1, actual.get({s, kj - 1}));
    expectConsistencyError = true;
}

TEST_F(TestDegreePrior, applyLabelMoveToDegreeCounts_forNonEmptyLabelMove_returnCorrectDegreeCounts){
    while(blockPrior.getBlockCount() == 1 || blockPrior.getBlockOfIdx(0) != 0) prior.sample();
    BlockMove move = {0, 0, 1};
    size_t k = prior.getState()[0], r = blockPrior.getBlockOfIdx(0);
    auto expected = prior.getDegreeCounts();
    prior.applyLabelMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCounts();
    EXPECT_EQ(expected.get({0, k}) - 1, actual.get({0, k}));
    EXPECT_EQ(expected.get({1, k}) + 1, actual.get({1, k}));
    expectConsistencyError = true;
}

TEST_F(TestDegreePrior, applyLabelMoveToDegreeCounts_forAddedLabelMove_returnCorrectDegreeCounts){
    while(blockPrior.getBlockCount() != 2 || blockPrior.getBlockOfIdx(0) != 0) prior.sample();
    BlockMove move = {0, 0, 2};
    size_t k = prior.getState()[0], r = 0, s = 2;
    auto expected = prior.getDegreeCounts();
    prior.applyLabelMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCounts();
    EXPECT_EQ(expected.get({0, k}) - 1, actual.get({0, k}));
    EXPECT_EQ(1, actual.get({2, k}));
    expectConsistencyError = true;
}


class TestDegreeUniformPrior: public ::testing::Test {
    public:

        BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
        BlockUniformPrior blockPrior = BlockUniformPrior(100, blockCountPrior);
        EdgeCountPoissonPrior edgeCountPrior = EdgeCountPoissonPrior(200);
        EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
        DegreeUniformPrior prior = DegreeUniformPrior(blockPrior, edgeMatrixPrior);
        void SetUp() {
            prior.sample();
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestDegreeUniformPrior, sampleState_returnConsistentState){
    prior.sampleState();
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestDegreeUniformPrior, getLogLikelihood_returnNonPositiveValue){
    double logLikelihood = prior.getLogLikelihood();
    EXPECT_LE(logLikelihood, 0);
}

TEST_F(TestDegreeUniformPrior, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectRatio){
    GraphMove move = {{}, {{0,1}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}

TEST_F(TestDegreeUniformPrior, getLogLikelihoodRatioFromLabelMove_forSomeLabelMove_returnCorrectRatio){
    BaseGraph::VertexIndex idx = 0;
    while (prior.getBlockPrior().getBlockCount() == 1) prior.sample();
    auto g = generateDCSBM(prior.getBlockPrior().getState(), prior.getEdgeMatrixPrior().getState().getAdjacencyMatrix(), prior.getState());
    prior.setGraph(g);
    BlockIndex prevBlockIdx = prior.getBlockPrior().getState()[idx];
    BlockIndex nextBlockIdx = prior.getBlockPrior().getState()[idx] + 1;
    if (nextBlockIdx == prior.getBlockPrior().getBlockCount())
        nextBlockIdx -= 2;
    BlockMove move = {idx, prevBlockIdx, nextBlockIdx};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}


class TestDegreeUniformHyperPrior: public ::testing::Test {
    public:

        BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
        BlockUniformPrior blockPrior = BlockUniformPrior(100, blockCountPrior);
        EdgeCountPoissonPrior edgeCountPrior = EdgeCountPoissonPrior(200);
        EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
        DegreeUniformHyperPrior prior = DegreeUniformHyperPrior(blockPrior, edgeMatrixPrior);
        void SetUp() {
            prior.sample();
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestDegreeUniformHyperPrior, sampleState_returnConsistentState){
    prior.sampleState();
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestDegreeUniformHyperPrior, getLogLikelihood_returnNonPositiveValue){
    double logLikelihood = prior.getLogLikelihood();
    EXPECT_LE(logLikelihood, 0);
}

TEST_F(TestDegreeUniformHyperPrior, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectRatio){
    GraphMove move = {{}, {{0,1}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyGraphMove(move);

    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}

TEST_F(TestDegreeUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forSomeLabelMove_returnCorrectRatio){
    BaseGraph::VertexIndex idx = 0;
    while (prior.getBlockPrior().getBlockCount() == 1) prior.sample();
    auto g = generateDCSBM(prior.getBlockPrior().getState(), prior.getEdgeMatrixPrior().getState().getAdjacencyMatrix(), prior.getState());
    prior.setGraph(g);
    BlockIndex prevBlockIdx = prior.getBlockPrior().getState()[idx];
    BlockIndex nextBlockIdx = prior.getBlockPrior().getState()[idx] + 1;
    if (nextBlockIdx == prior.getBlockPrior().getBlockCount())
        nextBlockIdx -= 2;
    BlockMove move = {idx, prevBlockIdx, nextBlockIdx};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}

TEST_F(TestDegreeUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forLabelMoveAddingNewBlock_returnCorrectRatio){
    BaseGraph::VertexIndex idx = 0;
    auto g = generateDCSBM(blockPrior.getState(), prior.getEdgeMatrixPrior().getState().getAdjacencyMatrix(), prior.getState());
    prior.setGraph(g);
    BlockMove move = {idx, blockPrior.getBlockOfIdx(idx), blockPrior.getVertexCounts().size()};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);

    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}

}
