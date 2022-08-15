#include "gtest/gtest.h"
#include <vector>
#include <iostream>


#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/degree.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "../fixtures.hpp"

using namespace FastMIDyNet;


namespace FastMIDyNet{

class DummyDegreePrior: public DegreePrior {
    public:
        DummyDegreePrior(size_t size, EdgeCountPrior& prior):
        DegreePrior(size, prior) {};

        void sampleState() override {
            DegreeSequence degreeSeq(getSize(), 0);
            degreeSeq[0] = getEdgeCount();
            degreeSeq[6] = getEdgeCount();
            setState(degreeSeq);
        }
        const double getLogLikelihood() const override { return 0; }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override { return 0; }
        void applyGraphMoveToState(const GraphMove& move) { DegreePrior::applyGraphMoveToState(move); }
        void applyGraphMoveToDegreeCounts(const GraphMove& move) { DegreePrior::applyGraphMoveToDegreeCounts(move); }


};

class DegreePriorTest: public ::testing::Test {
    public:

        double SIZE=10;
        double EDGE_COUNT=10;
        double TOL=1E-8;
        EdgeCountPoissonPrior edgeCountPrior = EdgeCountPoissonPrior(EDGE_COUNT);
        DummyDegreePrior prior = DummyDegreePrior(SIZE, edgeCountPrior);
        MultiGraph graph = getUndirectedHouseMultiGraph();

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

// TEST_F(DegreePriorTest, setGraph_forHouseGraph_applyChangesToDegreeSequence){
//     prior.setGraph(graph);
//     DegreeSequence expectedDegreeSeq = {4, 3, 5, 5, 2, 3, 0};
//     DegreeSequence actualDegreeSeq = prior.getState();
//     for (size_t i=0; i < 7; ++i){
//         EXPECT_EQ(expectedDegreeSeq[i], actualDegreeSeq[i]);
//     }
// }

// TEST_F(DegreePriorTest, computeDegreeCounts_forLocalDegreeSeqNBlockSeq_returnCorrectDegreeCounts){
//     blockPrior.setState({0,0,0,0,1,1,1});
//     auto degreeCounts = prior.computeDegreeCounts(prior.getState(), prior.getBlockPrior().getState());
//     EXPECT_EQ(degreeCounts.size(), 2);
//
//     EXPECT_EQ(degreeCounts[0].size(), 2);
//     EXPECT_FALSE(degreeCounts[0].isEmpty(0));
//     EXPECT_EQ(degreeCounts[0].get(0), 3);
//     EXPECT_FALSE(degreeCounts[0].isEmpty(prior.getEdgeMatrixPrior().getEdgeCount()));
//     EXPECT_EQ(degreeCounts[0].get(prior.getEdgeMatrixPrior().getEdgeCount()), 1);
//
//     EXPECT_EQ(degreeCounts[1].size(), 2);
//     EXPECT_FALSE(degreeCounts[1].isEmpty(0));
//     EXPECT_EQ(degreeCounts[1].get(0), 2);
//     EXPECT_FALSE(degreeCounts[1].isEmpty(prior.getEdgeMatrixPrior().getEdgeCount()));
//     EXPECT_EQ(degreeCounts[1].get(prior.getEdgeMatrixPrior().getEdgeCount()), 1);
// }

TEST_F(DegreePriorTest, applyGraphMoveToState_ForAddedEdge_returnCorrectDegreeSeq){
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

TEST_F(DegreePriorTest, applyGraphMoveToState_ForRemovedEdge_returnCorrectDegreeSeq){
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

TEST_F(DegreePriorTest, applyGraphMoveToState_ForRemovedEdgeNAddedEdge_returnCorrectDegreeSeq){
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

TEST_F(DegreePriorTest, applyGraphMoveToDegreeCounts_forAddedEdge_returnCorrectDegreeCounts){
    GraphMove move = {{}, {{0,1}}};
    auto expected = prior.getDegreeCounts();
    size_t k0 = prior.getDegreeOfIdx(0), k1 = prior.getDegreeOfIdx(1);
    expected.decrement(k0); expected.increment(k0 + 1);
    expected.decrement(k1); expected.increment(k1+1);
    prior.applyGraphMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCounts();

    // for (size_t r = 0; r < blockPrior.getBlockCount(); ++r){
    //     EXPECT_TRUE(expected[r] == actual[r]);
    // }
    for (auto nk : expected)
        EXPECT_EQ(nk.second, actual.get(nk.first));
    expectConsistencyError = true;

}

TEST_F(DegreePriorTest, applyGraphMoveToDegreeCounts_forRemovedEdge_returnCorrectDegreeCounts){
    prior.sample();
    GraphMove move = {{{0, 4}}, {}};
    size_t E = prior.getEdgeCount();
    auto expected = prior.getDegreeCounts();
    prior.applyGraphMoveToDegreeCounts(move);
    auto actual = prior.getDegreeCounts();

    size_t ki = prior.getState()[0],  kj = prior.getState()[4];

    EXPECT_EQ(expected.get(ki) - 1, actual.get(ki));
    EXPECT_EQ(expected.get(ki - 1) + 1, actual.get(ki - 1));

    EXPECT_EQ(expected.get(kj) - 1, actual.get(kj));
    EXPECT_EQ(expected.get(kj - 1) + 1, actual.get(kj - 1));
    expectConsistencyError = true;
}

// class TestDegreeUniformPrior: public ::testing::Test {
//     public:
//
//         BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
//         BlockUniformPrior blockPrior = BlockUniformPrior(100, blockCountPrior);
//         EdgeCountPoissonPrior edgeCountPrior = EdgeCountPoissonPrior(200);
//         EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
//         DegreeUniformPrior prior = DegreeUniformPrior(edgeMatrixPrior);
//         void SetUp() {
//             prior.sample();
//             prior.checkSafety();
//         }
//         void TearDown(){
//             prior.checkConsistency();
//         }
// };
//
// TEST_F(TestDegreeUniformPrior, sampleState_returnConsistentState){
//     prior.sampleState();
//     EXPECT_NO_THROW(prior.checkSelfConsistency());
// }
//
// TEST_F(TestDegreeUniformPrior, getLogLikelihood_returnNonPositiveValue){
//     double logLikelihood = prior.getLogLikelihood();
//     EXPECT_LE(logLikelihood, 0);
// }
//
// TEST_F(TestDegreeUniformPrior, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectRatio){
//     GraphMove move = {{}, {{0,1}}};
//     double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
//     double logLikelihoodBefore = prior.getLogLikelihood();
//     prior.applyGraphMove(move);
//     double logLikelihoodAfter = prior.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
// }
//
// TEST_F(TestDegreeUniformPrior, getLogLikelihoodRatioFromLabelMove_forSomeLabelMove_returnCorrectRatio){
//     BaseGraph::VertexIndex idx = 0;
//     while (prior.getBlockPrior().getBlockCount() == 1) prior.sample();
//     auto g = generateDCSBM(prior.getBlockPrior().getState(), prior.getEdgeMatrixPrior().getState().getAdjacencyMatrix(), prior.getState());
//     edgeMatrixPrior.setGraph(g);
//     BlockIndex prevBlockIdx = prior.getBlockPrior().getState()[idx];
//     BlockIndex nextBlockIdx = prior.getBlockPrior().getState()[idx] + 1;
//     if (nextBlockIdx == prior.getBlockPrior().getBlockCount())
//         nextBlockIdx -= 2;
//     BlockMove move = {idx, prevBlockIdx, nextBlockIdx};
//     double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = prior.getLogLikelihood();
//     prior.applyLabelMove(move);
//     double logLikelihoodAfter = prior.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
// }
//
//
// class TestDegreeUniformHyperPrior: public ::testing::Test {
//     public:
//
//         BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
//         BlockUniformPrior blockPrior = BlockUniformPrior(100, blockCountPrior);
//         EdgeCountPoissonPrior edgeCountPrior = EdgeCountPoissonPrior(200);
//         EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);
//         DegreeUniformHyperPrior prior = DegreeUniformHyperPrior(edgeMatrixPrior);
//         void SetUp() {
//             prior.sample();
//             prior.checkSafety();
//         }
//         void TearDown(){
//             prior.checkConsistency();
//         }
// };
//
// TEST_F(TestDegreeUniformHyperPrior, sampleState_returnConsistentState){
//     prior.sampleState();
// }
//
// TEST_F(TestDegreeUniformHyperPrior, getLogLikelihood_returnNonPositiveValue){
//     double logLikelihood = prior.getLogLikelihood();
//     EXPECT_LE(logLikelihood, 0);
// }
//
// TEST_F(TestDegreeUniformHyperPrior, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectRatio){
//     GraphMove move = {{}, {{0,1}}};
//     double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
//     double logLikelihoodBefore = prior.getLogLikelihood();
//     prior.applyGraphMove(move);
//
//     double logLikelihoodAfter = prior.getLogLikelihood();
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
// }
//
// TEST_F(TestDegreeUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forSomeLabelMove_returnCorrectRatio){
//     BaseGraph::VertexIndex idx = 0;
//     while (prior.getBlockPrior().getBlockCount() == 1) prior.sample();
//     auto g = generateDCSBM(prior.getBlockPrior().getState(), prior.getEdgeMatrixPrior().getState().getAdjacencyMatrix(), prior.getState());
//     edgeMatrixPrior.setGraph(g);
//     BlockIndex prevBlockIdx = prior.getBlockPrior().getState()[idx];
//     BlockIndex nextBlockIdx = prior.getBlockPrior().getState()[idx] + 1;
//     if (nextBlockIdx == prior.getBlockPrior().getBlockCount())
//         nextBlockIdx -= 2;
//     BlockMove move = {idx, prevBlockIdx, nextBlockIdx};
//     double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = prior.getLogLikelihood();
//     prior.applyLabelMove(move);
//     double logLikelihoodAfter = prior.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
// }
//
// TEST_F(TestDegreeUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forLabelMoveAddingNewBlock_returnCorrectRatio){
//     BaseGraph::VertexIndex idx = 0;
//     auto g = generateDCSBM(blockPrior.getState(), prior.getEdgeMatrixPrior().getState().getAdjacencyMatrix(), prior.getState());
//     BlockMove move = {idx, blockPrior.getBlockOfIdx(idx), blockPrior.getVertexCounts().size(), 1};
//     double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
//
//     double logLikelihoodBefore = prior.getLogLikelihood();
//     prior.applyLabelMove(move);
//     double logLikelihoodAfter = prior.getLogLikelihood();
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, TOL);
// }

}
