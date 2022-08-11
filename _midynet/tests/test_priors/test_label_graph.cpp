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

class LabelGraphPriorTest: public ::testing::Test {
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


TEST_F(LabelGraphPriorTest, setGraph_anyGraph_labelGraphCorrectlySet) {
    // EXPECT_EQ(prior.getState(), Matrix<size_t>({{8, 6}, {6, 2}}) );
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 4);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 6);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 1);
    EXPECT_EQ(prior.getEdgeCounts()[0], 14);
    EXPECT_EQ(prior.getEdgeCounts()[1], 8);
}

TEST_F(LabelGraphPriorTest, samplePriors_anyGraph_returnSumOfPriors) {
    double tmp = prior.getLogPrior();
    prior.computationFinished();
    EXPECT_EQ(tmp, edgeCountPrior.getLogJoint()+blockPrior.getLogJoint());
}

// TEST_F(LabelGraphPriorTest, createBlock_anySetup_addRowAndColumnToLabelGraph) {
//     prior.onBlockCreation({0, 0, 2, 1});
//     EXPECT_EQ(prior.getState(), Matrix<size_t>( {{8, 6, 0}, {6, 2, 0}, {0, 0, 0}} ));
//     expectConsistencyError = true;
// }
//
// TEST_F(LabelGraphPriorTest, createBlock_anySetup_addElementToEdgeCountOfBlocks) {
//     prior.onBlockCreation({0, 0, 2, 1});
//     EXPECT_EQ(prior.getEdgeCounts(), std::vector<size_t>({14, 8, 0}));
//     expectConsistencyError = true;
// }

TEST_F(LabelGraphPriorTest, applyLabelMoveToState_vertexChangingBlock_neighborsOfVertexChangedOfBlocks) {
    prior.applyLabelMoveToState({0, 0, 1});
    // EXPECT_EQ(prior.getState(), Matrix<size_t>({{6, 4}, {4, 8}}) );
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 3);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 4);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 4);

}

TEST_F(LabelGraphPriorTest, applyLabelMoveToState_vertexChangingBlockWithSelfloop_neighborsOfVertexChangedOfBlocks) {
    prior.applyLabelMoveToState({5, 1, 0});
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 6);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 5);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 0);
}
TEST_F(LabelGraphPriorTest, checkSelfConsistency_validData_noThrow) {
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

// TEST_F(LabelGraphPriorTest, checkSelfConsistency_incorrectBlockNumber_throwConsistencyError) {
//     blockCountPrior.setState(1);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     blockCountPrior.setState(3);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

// TEST_F(LabelGraphPriorTest, checkSelfConsistency_labelGraphNotOfBlockNumberSize_throwConsistencyError) {
//     MultiGraph labelGraph(3);
//     labelGraph.addEdgeIdx(0, 1);
//     labelGraph.addEdgeIdx(0, 2);
//     prior.setState(labelGraph);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

// TEST_F(LabelGraphPriorTest, checkSelfConsistency_incorrectEdgeNumber_throwConsistencyError) {
//     MultiGraph labelGraph(2);
//     labelGraph.addEdgeIdx(0, 1);
//     labelGraph.addEdgeIdx(1, 1);
//     labelGraph.addEdgeIdx(0, 0);
//     prior.setState(labelGraph);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

class LabelGraphDeltaPriorTest: public ::testing::Test {
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

TEST_F(LabelGraphDeltaPriorTest, sample_returnSameLabelGraph){
    MultiGraph labelGraph = prior.getState();
    prior.sample();
    EXPECT_EQ(labelGraph, prior.getState());
}

TEST_F(LabelGraphDeltaPriorTest, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0);
}

TEST_F(LabelGraphDeltaPriorTest, getLogLikelihoodRatioFromGraphMove_forGraphMovePreservingLabelGraph_return0){
    GraphMove move = {{}, {}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0);
}

TEST_F(LabelGraphDeltaPriorTest, getLogLikelihoodRatioFromGraphMove_forGraphMoveNotPreservingLabelGraph_returnMinusInfinity){
    GraphMove move = {{}, {{0, 1}}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), -INFINITY);
}

TEST_F(LabelGraphDeltaPriorTest, getLogLikelihoodRatioFromLabelMove_forLabelMovePreservingLabelGraph_return0){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0)};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0);
}

TEST_F(LabelGraphDeltaPriorTest, getLogLikelihoodRatioFromLabelMove_forLabelMoveNotPreservingLabelGraph_returnMinusInfinity){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), -INFINITY);
}

class LabelGraphErdosRenyiPriorTest: public ::testing::Test {
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

TEST_F(LabelGraphErdosRenyiPriorTest, sample_returnLabelGraphWithCorrectShape){
    prior.sample();
    EXPECT_EQ(prior.getState().getSize(), prior.getBlockPrior().getBlockCount());
    EXPECT_EQ(prior.getState().getTotalEdgeNumber(), prior.getEdgeCount());
}

TEST_F(LabelGraphErdosRenyiPriorTest, getLogLikelihood_forSomeSampledMatrix_returnCorrectLogLikelihood){
    prior.sample();
    auto E = prior.getEdgeCount(), B = prior.getBlockPrior().getEffectiveBlockCount();
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = -logMultisetCoefficient( B * (B + 1) / 2, E);
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelGraphErdosRenyiPriorTest, applyMove_forSomeGraphMove_changeLabelGraph){
    GraphMove move = {{{0, 0}}, {{0, 2}}};
    prior.applyGraphMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(LabelGraphErdosRenyiPriorTest, getLogLikelihoodRatio_forSomeGraphMoveContainingASelfLoop_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{{0, 0}}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelGraphErdosRenyiPriorTest, getLogLikelihoodRatio_forSomeGraphMoveChangingTheEdgeCount_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelGraphErdosRenyiPriorTest, applyMove_forSomeLabelMove_changeLabelGraph){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    prior.applyLabelMove(move);
}

TEST_F(LabelGraphErdosRenyiPriorTest, getLogLikelihoodRatio_forSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelGraphErdosRenyiPriorTest, getLogLikelihoodRatio_forSomeLabelMoveAddingNewBlock_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockCount(), 1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);

}

TEST_F(LabelGraphErdosRenyiPriorTest, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

// TEST_F(LabelGraphErdosRenyiPriorTest, checkSelfConsistency_inconsistenBlockCount_throwConsistencyError){
//     size_t originalBlockCount = blockCountPrior.getState();
//     blockCountPrior.setState(10);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     blockCountPrior.setState(originalBlockCount);
// }

TEST_F(LabelGraphErdosRenyiPriorTest, checkSelfConsistency_inconsistentEdgeCount_throwConsistencyError){
    size_t originalEdgeCount = edgeCountPrior.getState();
    edgeCountPrior.setState(50);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    edgeCountPrior.setState(originalEdgeCount);
}


class LabelPlantedPartitionPriorTest: public ::testing::Test {
    public:
        const BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};
        MultiGraph graph = getUndirectedHouseMultiGraph();
        EdgeCountDeltaPrior edgeCountPrior = {11};
        BlockCountDeltaPrior blockCountPrior = {3};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        LabelGraphPlantedPartitionPrior prior = {edgeCountPrior, blockPrior};

        bool expectConsistencyError = false;
        void SetUp() {
            seedWithTime();
            blockPrior.setState(BLOCK_SEQUENCE);
            prior.setGraph(graph);
            prior.checkSafety();
        }
        void TearDown(){
            if (not expectConsistencyError)
                prior.checkConsistency();
        }
};

TEST_F(LabelPlantedPartitionPriorTest, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(LabelPlantedPartitionPriorTest, sample_returnLabelGraphWithCorrectShape){
    prior.sample();
    EXPECT_EQ(prior.getState().getSize(), prior.getBlockPrior().getBlockCount());
    EXPECT_EQ(prior.getState().getTotalEdgeNumber(), prior.getEdgeCount());
    expectConsistencyError = true;
}

TEST_F(LabelPlantedPartitionPriorTest, getLogLikelihood_forSomeSampledMatrix_returnCorrectLogLikelihood){
    prior.sample();
    auto E = prior.getEdgeCount(), B = prior.getBlockPrior().getEffectiveBlockCount();
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = - ((B == 1) ? 0 : log(E + 1)) + logFactorial(prior.getEdgeCountIn())  + logFactorial(prior.getEdgeCountOut());
    expectedLogLikelihood -= log(B) * prior.getEdgeCountIn() + log(B * (B - 1) / 2) * prior.getEdgeCountOut();
    for(size_t r=0; r<prior.getState().getSize(); ++r){
        expectedLogLikelihood -= logFactorial(prior.getState().getEdgeMultiplicityIdx(r, r));
        for(size_t s=r+1; s<prior.getState().getSize(); ++s){
            expectedLogLikelihood -= logFactorial(prior.getState().getEdgeMultiplicityIdx(r, s));
        }
    }
    EXPECT_NEAR(actualLogLikelihood, expectedLogLikelihood, 1e-6);
    expectConsistencyError = true;
}

TEST_F(LabelPlantedPartitionPriorTest, applyMove_forSomeGraphMove_changeLabelGraph){
    GraphMove move = {{{0, 2}}, {{0, 0}}};
    prior.applyGraphMove(move);
    graph.removeEdgeIdx(0, 2);
    graph.addEdgeIdx(0, 0);
    EXPECT_NO_THROW(prior.checkSelfConsistency());

}

TEST_F(LabelPlantedPartitionPriorTest, getLogLikelihoodRatio_forSomeGraphMoveContainingASelfLoop_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{{0, 2}}, {{0, 0}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    graph.removeEdgeIdx(0, 2);
    graph.addEdgeIdx(0, 0);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelPlantedPartitionPriorTest, getLogLikelihoodRatio_forSomeGraphMoveChangingTheEdgeCount_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    graph.addEdgeIdx(0, 0);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelPlantedPartitionPriorTest, applyMove_forSomeLabelMove_changeLabelGraph){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    prior.applyLabelMove(move);
}

TEST_F(LabelPlantedPartitionPriorTest, getLogLikelihoodRatio_forSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

TEST_F(LabelPlantedPartitionPriorTest, getLogLikelihoodRatio_forSomeLabelMoveAddingNewBlock_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockCount(), 1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_NEAR(actualLogLikelihoodRatio, expectedLogLikelihood, 1e-6);
}

}
