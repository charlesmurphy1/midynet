#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
// #include "../fixtures.hpp"


namespace FastMIDyNet{


static FastMIDyNet::MultiGraph getUndirectedHouseMultiGraph(){
    //     /*
    //      * (0)     (1)
    //      * |||\   / | \
    //      * ||| \ /  |  \
    //      * |||  X   |  (4)
    //      * ||| / \  |  /
    //      * |||/   \ | /
    //      * (2)-----(3)-----(5)--
    //      *                   \__|
    //      *      (6)
    //      */
    // k = {4, 3, 5, 5, 2, 3, 0}
    FastMIDyNet::MultiGraph graph(7);
    graph.addMultiedgeIdx(0, 2, 3);
    graph.addEdgeIdx(0, 3);
    graph.addEdgeIdx(1, 2);
    graph.addEdgeIdx(1, 3);
    graph.addEdgeIdx(1, 4);
    graph.addEdgeIdx(2, 3);
    graph.addEdgeIdx(3, 4);
    graph.addEdgeIdx(3, 5);
    graph.addEdgeIdx(5, 5);

    return graph;

}


class DummyLabelGraphPrior: public LabelGraphPrior {
    void _applyGraphMove(const GraphMove&) override { };
    void _applyLabelMove(const BlockMove&) override { };
    public:
        using LabelGraphPrior::LabelGraphPrior;
        void sampleState() {}
        const double getLogLikelihood() const override { return 0.; }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override { return 0; }
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const override { return 0; }

        void applyLabelMoveToState(const BlockMove& move) { LabelGraphPrior::applyLabelMoveToState(move); }
};

class TestLabelGraphPrior: public ::testing::Test {
    public:
        const BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};
        MultiGraph graph = getUndirectedHouseMultiGraph();
        EdgeCountPoissonPrior edgeCountPrior = {2};
        BlockCountPoissonPrior blockCountPrior = {2};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        bool expectConsistencyError = false;

        DummyLabelGraphPrior prior = {edgeCountPrior, blockPrior};

        void SetUp() {
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());

            prior.setGraph(graph);
            prior.checkSafety();
        }
        void TearDown(){
            if (not expectConsistencyError)
                prior.checkConsistency();
        }
};


TEST_F(TestLabelGraphPrior, setGraph_anyGraph_labelGraphCorrectlySet) {
    // EXPECT_EQ(prior.getState(), Matrix<size_t>({{8, 6}, {6, 2}}) );
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 4);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 6);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 1);
    EXPECT_EQ(prior.getEdgeCounts()[0], 14);
    EXPECT_EQ(prior.getEdgeCounts()[1], 8);
}

TEST_F(TestLabelGraphPrior, samplePriors_anyGraph_returnSumOfPriors) {
    double tmp = prior.getLogPrior();
    prior.computationFinished();
    EXPECT_EQ(tmp, edgeCountPrior.getLogJoint()+blockPrior.getLogJoint());
}

// TEST_F(TestLabelGraphPrior, createBlock_anySetup_addRowAndColumnToLabelGraph) {
//     prior.onBlockCreation({0, 0, 2, 1});
//     EXPECT_EQ(prior.getState(), Matrix<size_t>( {{8, 6, 0}, {6, 2, 0}, {0, 0, 0}} ));
//     expectConsistencyError = true;
// }
//
// TEST_F(TestLabelGraphPrior, createBlock_anySetup_addElementToEdgeCountOfBlocks) {
//     prior.onBlockCreation({0, 0, 2, 1});
//     EXPECT_EQ(prior.getEdgeCounts(), std::vector<size_t>({14, 8, 0}));
//     expectConsistencyError = true;
// }

TEST_F(TestLabelGraphPrior, applyLabelMoveToState_vertexChangingBlock_neighborsOfVertexChangedOfBlocks) {
    prior.applyLabelMoveToState({0, 0, 1});
    // EXPECT_EQ(prior.getState(), Matrix<size_t>({{6, 4}, {4, 8}}) );
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 3);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 4);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 4);

}

TEST_F(TestLabelGraphPrior, applyLabelMoveToState_vertexChangingBlockWithSelfloop_neighborsOfVertexChangedOfBlocks) {
    prior.applyLabelMoveToState({5, 1, 0});
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 6);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 5);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 0);
}
TEST_F(TestLabelGraphPrior, checkSelfConsistency_validData_noThrow) {
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

// TEST_F(TestLabelGraphPrior, checkSelfConsistency_incorrectBlockNumber_throwConsistencyError) {
//     blockCountPrior.setState(1);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     blockCountPrior.setState(3);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

// TEST_F(TestLabelGraphPrior, checkSelfConsistency_labelGraphNotOfBlockNumberSize_throwConsistencyError) {
//     MultiGraph labelGraph(3);
//     labelGraph.addEdgeIdx(0, 1);
//     labelGraph.addEdgeIdx(0, 2);
//     prior.setState(labelGraph);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

// TEST_F(TestLabelGraphPrior, checkSelfConsistency_incorrectEdgeNumber_throwConsistencyError) {
//     MultiGraph labelGraph(2);
//     labelGraph.addEdgeIdx(0, 1);
//     labelGraph.addEdgeIdx(1, 1);
//     labelGraph.addEdgeIdx(0, 0);
//     prior.setState(labelGraph);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

class TestLabelGraphDeltaPrior: public ::testing::Test {
    public:
        const BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};
        MultiGraph graph = getUndirectedHouseMultiGraph();
        MultiGraph labelGraph = MultiGraph(2); // = {{10, 2}, {2, 10}};
        EdgeCountDeltaPrior edgeCountPrior = {7};
        BlockCountPoissonPrior blockCountPrior = {2};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        LabelGraphDeltaPrior prior = {labelGraph, edgeCountPrior, blockPrior};

        void SetUp() {
            labelGraph.addMultiedgeIdx(0, 0, 10);
            labelGraph.addMultiedgeIdx(0, 1, 2);
            labelGraph.addMultiedgeIdx(1, 1, 10);
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());
            prior.setGraph(graph);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestLabelGraphDeltaPrior, sample_returnSameLabelGraph){
    MultiGraph labelGraph = prior.getState();
    prior.sample();
    EXPECT_EQ(labelGraph, prior.getState());
}

TEST_F(TestLabelGraphDeltaPrior, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0);
}

TEST_F(TestLabelGraphDeltaPrior, getLogLikelihoodRatioFromGraphMove_forGraphMovePreservingLabelGraph_return0){
    GraphMove move = {{}, {}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0);
}

TEST_F(TestLabelGraphDeltaPrior, getLogLikelihoodRatioFromGraphMove_forGraphMoveNotPreservingLabelGraph_returnMinusInfinity){
    GraphMove move = {{}, {{0, 1}}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), -INFINITY);
}

TEST_F(TestLabelGraphDeltaPrior, getLogLikelihoodRatioFromLabelMove_forLabelMovePreservingLabelGraph_return0){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0)};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0);
}

TEST_F(TestLabelGraphDeltaPrior, getLogLikelihoodRatioFromLabelMove_forLabelMoveNotPreservingLabelGraph_returnMinusInfinity){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), -INFINITY);
}

class TestLabelGraphErdosRenyiPrior: public ::testing::Test {
    public:
        const BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};
        MultiGraph graph = getUndirectedHouseMultiGraph();
        EdgeCountDeltaPrior edgeCountPrior = {10};
        BlockCountDeltaPrior blockCountPrior = {3};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        LabelGraphErdosRenyiPrior prior = {edgeCountPrior, blockPrior};

        void SetUp() {
            seedWithTime();
            blockPrior.setState(BLOCK_SEQUENCE);
            prior.setGraph(graph);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestLabelGraphErdosRenyiPrior, sample_returnLabelGraphWithCorrectShape){
    prior.sample();
    EXPECT_EQ(prior.getState().getSize(), prior.getBlockPrior().getBlockCount());
    EXPECT_EQ(prior.getState().getTotalEdgeNumber(), prior.getEdgeCount());
}

TEST_F(TestLabelGraphErdosRenyiPrior, getLogLikelihood_forSomeSampledMatrix_returnCorrectLogLikelihood){
    prior.sample();
    auto E = prior.getEdgeCount(), B = prior.getBlockPrior().getEffectiveBlockCount();
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = -logMultisetCoefficient( B * (B + 1) / 2, E);
    EXPECT_EQ(actualLogLikelihood, expectedLogLikelihood);
}

TEST_F(TestLabelGraphErdosRenyiPrior, applyMove_forSomeGraphMove_changeLabelGraph){
    GraphMove move = {{{0, 0}}, {{0, 2}}};
    prior.applyGraphMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestLabelGraphErdosRenyiPrior, getLogLikelihoodRatio_forSomeGraphMoveContainingASelfLoop_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{{0, 0}}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);
}

TEST_F(TestLabelGraphErdosRenyiPrior, getLogLikelihoodRatio_forSomeGraphMoveChangingTheEdgeCount_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);
}

TEST_F(TestLabelGraphErdosRenyiPrior, applyMove_forSomeLabelMove_changeLabelGraph){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    prior.applyLabelMove(move);
}

TEST_F(TestLabelGraphErdosRenyiPrior, getLogLikelihoodRatio_forSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);

}

TEST_F(TestLabelGraphErdosRenyiPrior, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

// TEST_F(TestLabelGraphErdosRenyiPrior, checkSelfConsistency_inconsistenBlockCount_throwConsistencyError){
//     size_t originalBlockCount = blockCountPrior.getState();
//     blockCountPrior.setState(10);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     blockCountPrior.setState(originalBlockCount);
// }

TEST_F(TestLabelGraphErdosRenyiPrior, checkSelfConsistency_inconsistentEdgeCount_throwConsistencyError){
    size_t originalEdgeCount = edgeCountPrior.getState();
    edgeCountPrior.setState(50);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    edgeCountPrior.setState(originalEdgeCount);
}

}
