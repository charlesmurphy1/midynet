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
        const double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const { return 0; }

        void _createBlock(){ createBlock(); }
        void _destroyBlock(const BlockIndex& idx){ destroyBlock(idx); }

};

class TestDegreePrior: public ::testing::Test {
    public:

        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = FastMIDyNet::BlockCountPoissonPrior(POISSON_MEAN);
        FastMIDyNet::BlockUniformPrior blockPrior = FastMIDyNet::BlockUniformPrior(GRAPH_SIZE, blockCountPrior);
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = FastMIDyNet::EdgeCountPoissonPrior(POISSON_MEAN);
        FastMIDyNet::EdgeMatrixUniformPrior edgeMatrixPrior = FastMIDyNet::EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
        DummyDegreePrior prior = DummyDegreePrior(blockPrior, edgeMatrixPrior);
        void SetUp() {
            prior.sample();
            prior.computationFinished();
        }
};

TEST_F(TestDegreePrior, setGraph_forHouseGraph_applyChangesToDegreeSequence){
    prior.setGraph(getUndirectedHouseMultiGraph());
    FastMIDyNet::DegreeSequence expectedDegreeSeq = {4, 3, 5, 5, 2, 3, 0};
    FastMIDyNet::DegreeSequence actualDegreeSeq = prior.getState();
    for (size_t i=0; i < 7; ++i){
        EXPECT_EQ(expectedDegreeSeq[i], actualDegreeSeq[i]);
    }
}

TEST_F(TestDegreePrior, computeDegreeCountsInBlocks_forLocalDegreeSeqNBlockSeq_returnCorrectDegreeCountsInBlocks){
    blockPrior.setState({0,0,0,0,1,1,1});
    auto degreeCountsInBlocks = prior.computeDegreeCountsInBlocks(prior.getState(), prior.getBlockPrior().getState());
    EXPECT_EQ(degreeCountsInBlocks.size(), 2);

    EXPECT_EQ(degreeCountsInBlocks[0].size(), 2);
    EXPECT_FALSE(degreeCountsInBlocks[0].isEmpty(0));
    EXPECT_EQ(degreeCountsInBlocks[0].get(0), 3);
    EXPECT_FALSE(degreeCountsInBlocks[0].isEmpty(prior.getEdgeMatrixPrior().getEdgeCount()));
    EXPECT_EQ(degreeCountsInBlocks[0].get(prior.getEdgeMatrixPrior().getEdgeCount()), 1);

    EXPECT_EQ(degreeCountsInBlocks[1].size(), 2);
    EXPECT_FALSE(degreeCountsInBlocks[1].isEmpty(0));
    EXPECT_EQ(degreeCountsInBlocks[1].get(0), 2);
    EXPECT_FALSE(degreeCountsInBlocks[1].isEmpty(prior.getEdgeMatrixPrior().getEdgeCount()));
    EXPECT_EQ(degreeCountsInBlocks[1].get(prior.getEdgeMatrixPrior().getEdgeCount()), 1);
}

TEST_F(TestDegreePrior, applyGraphMoveToState_ForAddedEdge_returnCorrectDegreeSeq){
    FastMIDyNet::GraphMove move = {{}, {{0,1}}};
    auto k0Before = prior.getState()[0];
    auto k1Before = prior.getState()[1];
    prior.applyGraphMoveToState(move);
    auto k0After = prior.getState()[0];
    auto k1After = prior.getState()[1];
    EXPECT_EQ(k0After, k0Before + 1);
    EXPECT_EQ(k1After, k1Before + 1);
}

TEST_F(TestDegreePrior, applyGraphMoveToState_ForRemovedEdge_returnCorrectDegreeSeq){
    FastMIDyNet::GraphMove move = {{{0,6}}, {}};
    auto k0Before = prior.getState()[0];
    auto k6Before = prior.getState()[6];
    prior.applyGraphMoveToState(move);
    auto k0After = prior.getState()[0];
    auto k6After = prior.getState()[6];
    EXPECT_EQ(k0After, k0Before - 1);
    EXPECT_EQ(k6After, k6Before - 1);
}

TEST_F(TestDegreePrior, applyGraphMoveToState_ForRemovedEdgeNAddedEdge_returnCorrectDegreeSeq){
    FastMIDyNet::GraphMove move = {{{0,6}}, {{0, 1}}};
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
}

TEST_F(TestDegreePrior, applyBlockMoveToState_forNonEmptyBlockMove_doNothing){
    FastMIDyNet::BlockMove move = {0, prior.getBlockPrior().getState()[0], 0, 0};
    FastMIDyNet::DegreeSequence degreeSeqBefore = prior.getState();
    prior.applyBlockMoveToState(move);
    FastMIDyNet::DegreeSequence degreeSeqAfter = prior.getState();

    EXPECT_EQ(degreeSeqBefore.size(), degreeSeqAfter.size());
    for(size_t i=0; i<degreeSeqBefore.size(); ++i){
        EXPECT_EQ(degreeSeqBefore[i], degreeSeqAfter[i]);
    }
}

TEST_F(TestDegreePrior, applyGraphMoveToDegreeCounts_forAddedEdge_returnCorrectDegreeCounts){
    blockPrior.setState(BLOCK_SEQ);
    FastMIDyNet::GraphMove move = {{}, {{0,1}}};
    size_t E = prior.getEdgeMatrixPrior().getEdgeCount();
    auto expected = prior.getDegreeCountsInBlocks();
    expected[0].decrement(0); expected[0].increment(1);
    expected[0].decrement(E); expected[0].increment(E+1);
    prior.applyGraphMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCountsInBlocks();

    for (size_t r = 0; r < prior.getBlockPrior().getBlockCount(); ++r){
        EXPECT_TRUE(expected[r] == actual[r]);
    }
}

TEST_F(TestDegreePrior, applyGraphMoveToDegreeCounts_forRemovedEdge_returnCorrectDegreeCounts){
    blockPrior.setState(BLOCK_SEQ);
    FastMIDyNet::GraphMove move = {{{0, 6}}, {}};
    size_t E = prior.getEdgeMatrixPrior().getEdgeCount();
    auto expected = prior.getDegreeCountsInBlocks();
    expected[0].decrement(E); expected[0].increment(E-1);
    expected[1].decrement(E); expected[1].increment(E-1);
    prior.applyGraphMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCountsInBlocks();

    for (size_t r = 0; r < prior.getBlockPrior().getBlockCount(); ++r){
        EXPECT_TRUE(expected[r] == actual[r]);
    }
}

TEST_F(TestDegreePrior, applyBlockMoveToDegreeCounts_forNonEmptyBlockMove_returnCorrectDegreeCounts){
    FastMIDyNet::BlockMove move = {0, 0, 1, 0};
    while(prior.getBlockPrior().getBlockCount() == 1) prior.sample();
    size_t E = prior.getEdgeMatrixPrior().getEdgeCount();
    auto expected = prior.getDegreeCountsInBlocks();
    expected[0].decrement(E);;
    expected[1].increment(E);;
    prior.applyBlockMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCountsInBlocks();

    for (size_t r = 0; r < prior.getBlockPrior().getBlockCount(); ++r){
        EXPECT_TRUE(expected[r] == actual[r]);
    }
}



class TestDegreeUniformPrior: public ::testing::Test {
    public:

        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = FastMIDyNet::BlockCountPoissonPrior(POISSON_MEAN);
        FastMIDyNet::BlockUniformPrior blockPrior = FastMIDyNet::BlockUniformPrior(100, blockCountPrior);
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = FastMIDyNet::EdgeCountPoissonPrior(200);
        FastMIDyNet::EdgeMatrixUniformPrior edgeMatrixPrior = FastMIDyNet::EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
        FastMIDyNet::DegreeUniformPrior prior = FastMIDyNet::DegreeUniformPrior(blockPrior, edgeMatrixPrior);
        void SetUp() {
            prior.sample();
            prior.computationFinished();
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
    FastMIDyNet::GraphMove move = {{}, {{0,1}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}

TEST_F(TestDegreeUniformPrior, getLogLikelihoodRatioFromBlockMove_forSomeBlockMove_returnCorrectRatio){
    BaseGraph::VertexIndex idx = 0;
    auto g = FastMIDyNet::generateDCSBM(prior.getBlockPrior().getState(), prior.getEdgeMatrixPrior().getState(), prior.getState());
    prior.setGraph(g);
    FastMIDyNet::BlockMove move = {idx, prior.getBlockPrior().getState()[idx], prior.getBlockPrior().getState()[idx] + 1, 0};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromBlockMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();
    prior.applyBlockMove(move);
    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
}
