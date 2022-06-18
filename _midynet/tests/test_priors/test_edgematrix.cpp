#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "fixtures.hpp"


namespace FastMIDyNet{

const BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};


class DummyEdgeMatrixPrior: public EdgeMatrixPrior {
    public:
        using EdgeMatrixPrior::EdgeMatrixPrior;
        void sampleState() {}
        const double getLogLikelihood() const { return 0.; }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const { return 0; }
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const { return 0; }

        void applyGraphMove(const GraphMove&) { };
        void applyLabelMove(const BlockMove&) { };
        void applyLabelMoveToState(const BlockMove& move) { EdgeMatrixPrior::applyLabelMoveToState(move); }
};

class TestEdgeMatrixPrior: public ::testing::Test {
    public:
        MultiGraph graph = getUndirectedHouseMultiGraph();
        EdgeCountPoissonPrior edgeCountPrior = {2};
        BlockCountPoissonPrior blockCountPrior = {2};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        bool expectConsistencyError = false;

        DummyEdgeMatrixPrior prior = {edgeCountPrior, blockPrior};

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


TEST_F(TestEdgeMatrixPrior, setGraph_anyGraph_edgeMatrixCorrectlySet) {
    // EXPECT_EQ(prior.getState(), Matrix<size_t>({{8, 6}, {6, 2}}) );
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 4);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 6);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 1);
    EXPECT_EQ(prior.getEdgeCounts()[0], 14);
    EXPECT_EQ(prior.getEdgeCounts()[1], 8);
}

TEST_F(TestEdgeMatrixPrior, samplePriors_anyGraph_returnSumOfPriors) {
    double tmp = prior.getLogPrior();
    prior.computationFinished();
    EXPECT_EQ(tmp, edgeCountPrior.getLogJoint()+blockPrior.getLogJoint());
}

// TEST_F(TestEdgeMatrixPrior, createBlock_anySetup_addRowAndColumnToEdgeMatrix) {
//     prior.onBlockCreation({0, 0, 2, 1});
//     EXPECT_EQ(prior.getState(), Matrix<size_t>( {{8, 6, 0}, {6, 2, 0}, {0, 0, 0}} ));
//     expectConsistencyError = true;
// }
//
// TEST_F(TestEdgeMatrixPrior, createBlock_anySetup_addElementToEdgeCountOfBlocks) {
//     prior.onBlockCreation({0, 0, 2, 1});
//     EXPECT_EQ(prior.getEdgeCounts(), std::vector<size_t>({14, 8, 0}));
//     expectConsistencyError = true;
// }

TEST_F(TestEdgeMatrixPrior, applyLabelMoveToState_vertexChangingBlock_neighborsOfVertexChangedOfBlocks) {
    prior.applyLabelMoveToState({0, 0, 1});
    // EXPECT_EQ(prior.getState(), Matrix<size_t>({{6, 4}, {4, 8}}) );
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 3);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 4);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 4);

}

TEST_F(TestEdgeMatrixPrior, applyLabelMoveToState_vertexChangingBlockWithSelfloop_neighborsOfVertexChangedOfBlocks) {
    prior.applyLabelMoveToState({5, 1, 0});
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 0), 6);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(0, 1), 5);
    EXPECT_EQ(prior.getState().getEdgeMultiplicityIdx(1, 1), 0);
}
TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_validData_noThrow) {
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

// TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_incorrectBlockNumber_throwConsistencyError) {
//     blockCountPrior.setState(1);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     blockCountPrior.setState(3);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

// TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_edgeMatrixNotOfBlockNumberSize_throwConsistencyError) {
//     MultiGraph edgeMatrix(3);
//     edgeMatrix.addEdgeIdx(0, 1);
//     edgeMatrix.addEdgeIdx(0, 2);
//     prior.setState(edgeMatrix);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

// TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_incorrectEdgeNumber_throwConsistencyError) {
//     MultiGraph edgeMatrix(2);
//     edgeMatrix.addEdgeIdx(0, 1);
//     edgeMatrix.addEdgeIdx(1, 1);
//     edgeMatrix.addEdgeIdx(0, 0);
//     prior.setState(edgeMatrix);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     expectConsistencyError = true;
// }

class TestEdgeMatrixDeltaPrior: public ::testing::Test {
    public:
        MultiGraph graph = getUndirectedHouseMultiGraph();
        MultiGraph edgeMatrix = MultiGraph(2); // = {{10, 2}, {2, 10}};
        EdgeCountDeltaPrior edgeCountPrior = {7};
        BlockCountPoissonPrior blockCountPrior = {2};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        EdgeMatrixDeltaPrior prior = {edgeMatrix, edgeCountPrior, blockPrior};

        void SetUp() {
            edgeMatrix.addMultiedgeIdx(0, 0, 10);
            edgeMatrix.addMultiedgeIdx(0, 1, 2);
            edgeMatrix.addMultiedgeIdx(1, 1, 10);
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());
            prior.setGraph(graph);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestEdgeMatrixDeltaPrior, sample_returnSameEdgeMatrix){
    MultiGraph edgeMatrix = prior.getState();
    prior.sample();
    EXPECT_EQ(edgeMatrix, prior.getState());
}

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0);
}

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihoodRatioFromGraphMove_forGraphMovePreservingEdgeMatrix_return0){
    GraphMove move = {{}, {}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0);
}

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihoodRatioFromGraphMove_forGraphMoveNotPreservingEdgeMatrix_returnMinusInfinity){
    GraphMove move = {{}, {{0, 1}}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), -INFINITY);
}

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihoodRatioFromLabelMove_forLabelMovePreservingEdgeMatrix_return0){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0)};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0);
}

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihoodRatioFromLabelMove_forLabelMoveNotPreservingEdgeMatrix_returnMinusInfinity){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), -INFINITY);
}

class TestEdgeMatrixUniformPrior: public ::testing::Test {
    public:
        MultiGraph graph = getUndirectedHouseMultiGraph();
        EdgeCountPoissonPrior edgeCountPrior = {2};
        BlockCountPoissonPrior blockCountPrior = {2};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};

        EdgeMatrixUniformPrior prior = {edgeCountPrior, blockPrior};

        void SetUp() {
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());
            prior.setGraph(graph);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestEdgeMatrixUniformPrior, sample_returnEdgeMatrixWithCorrectShape){
    prior.sample();
    auto blockSeq = prior.getState();
    EXPECT_EQ(prior.getState().getSize(), prior.getBlockPrior().getBlockCount());

    auto sum = 0;
    for (auto r : prior.getState()){
        EXPECT_EQ(prior.getState().getNeighboursOfIdx(r).size(), prior.getBlockPrior().getBlockCount());
        for (auto s : prior.getState().getNeighboursOfIdx(r)){
            EXPECT_TRUE(s.label >= 0);
            sum += s.label;
        }
    }
    EXPECT_EQ(sum, 2 * prior.getEdgeCount());
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihood_forSomeSampledMatrix_returnCorrectLogLikelihood){
    prior.sample();
    auto E = prior.getEdgeCount(), B = prior.getBlockPrior().getBlockCount();
    double actualLogLikelihood = prior.getLogLikelihood();
    double expectedLogLikelihood = -logMultisetCoefficient( B * (B + 1) / 2, E);
    EXPECT_EQ(actualLogLikelihood, expectedLogLikelihood);
}

TEST_F(TestEdgeMatrixUniformPrior, applyMove_forSomeGraphMove_changeEdgeMatrix){
    GraphMove move = {{{0, 0}}, {{0, 2}}};
    prior.applyGraphMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeGraphMoveContainingASelfLoop_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{{0, 0}}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeGraphMoveChangingTheEdgeCount_returnCorrectLogLikelihoodRatio){
    GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);
}

TEST_F(TestEdgeMatrixUniformPrior, applyMove_forSomeLabelMove_changeEdgeMatrix){
    BlockMove move = {0, BLOCK_SEQUENCE[0], BLOCK_SEQUENCE[0] + 1};
    prior.applyLabelMove(move);
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, BLOCK_SEQUENCE[0], BLOCK_SEQUENCE[0]+1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyLabelMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);

}

TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

// TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_inconsistenBlockCount_throwConsistencyError){
//     size_t originalBlockCount = blockCountPrior.getState();
//     blockCountPrior.setState(10);
//     EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
//     blockCountPrior.setState(originalBlockCount);
// }

TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_inconsistentEdgeCount_throwConsistencyError){
    size_t originalEdgeCount = edgeCountPrior.getState();
    edgeCountPrior.setState(50);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    edgeCountPrior.setState(originalEdgeCount);
}

}
