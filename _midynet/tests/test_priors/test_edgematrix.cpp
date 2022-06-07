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
        const double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const { return 0; }

        void applyGraphMove(const GraphMove&) { };
        void applyBlockMove(const BlockMove&) { };

        void _moveEdgeCountsInBlocks(const BlockMove& move) { moveEdgeCountsInBlocks(move); }
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
    EXPECT_EQ(prior.getState(),
            Matrix<size_t>({{8, 6}, {6, 2}})
        );
    EXPECT_EQ(prior.getEdgeCountsInBlocks(), BlockSequence({14, 8}));
}

TEST_F(TestEdgeMatrixPrior, samplePriors_anyGraph_returnSumOfPriors) {
    double tmp = prior.getLogPrior();
    prior.computationFinished();
    EXPECT_EQ(tmp, edgeCountPrior.getLogJoint()+blockPrior.getLogJoint());
}

TEST_F(TestEdgeMatrixPrior, createBlock_anySetup_addRowAndColumnToEdgeMatrix) {
    prior.createBlock();
    EXPECT_EQ(prior.getState(),
            Matrix<size_t>( {{8, 6, 0}, {6, 2, 0}, {0, 0, 0}} ));
    expectConsistencyError = true;

}

TEST_F(TestEdgeMatrixPrior, createBlock_anySetup_addElementToEdgeCountOfBlocks) {
    prior.createBlock();
    EXPECT_EQ(prior.getEdgeCountsInBlocks(), std::vector<size_t>({14, 8, 0}));
    expectConsistencyError = true;
}

// TEST_F(TestEdgeMatrixPrior, destroyBlock_anyBlock_removeFirstRowAndColumn) {
//     size_t removedBlock = 0;
//
//     for (const BlockSequence& blockSequence:
//             std::list<BlockSequence>{{1, 1, 1, 1, 1, 2, 1}, {0, 0, 0, 0, 0, 2, 0}}) {
//
//         blockPrior.setState(blockSequence);
//         // blockCountPrior.setState(3); // blockPrior setState changes blockCountPrior state
//         prior.setGraph(graph);
//
//         std::cout << "Before destroying block: #blocks = " << blockPrior.getBlockCount() << std::endl;
//         prior.destroyBlock(removedBlock);
//         std::cout << "After destroying block: #blocks = " << blockPrior.getBlockCount() << std::endl;
//         EXPECT_EQ(prior.getState(),
//                 Matrix<size_t>( {{18, 1}, {1, 2}} ));
//         EXPECT_EQ(prior.getEdgeCountsInBlocks(), std::vector<size_t>({19, 3}));
//
//         removedBlock++;
//     }
// }

TEST_F(TestEdgeMatrixPrior, moveEdgesInBlocks_vertexChangingBlock_neighborsOfVertexChangedOfBlocks) {
    prior._moveEdgeCountsInBlocks({0, 0, 1});
    EXPECT_EQ(prior.getState(),
            Matrix<size_t>({{6, 4}, {4, 8}})
        );
}

TEST_F(TestEdgeMatrixPrior, moveEdgesInBlocks_vertexChangingBlockWithSelfloop_neighborsOfVertexChangedOfBlocks) {
    prior._moveEdgeCountsInBlocks({5, 1, 0});
    EXPECT_EQ(prior.getState(),
            Matrix<size_t>({{12, 5}, {5, 0}})
        );
}
TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_validData_noThrow) {
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_incorrectBlockNumber_throwConsistencyError) {
    blockCountPrior.setState(1);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    blockCountPrior.setState(3);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    expectConsistencyError = true;

}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_edgeMatrixNotOfBlockNumberSize_throwConsistencyError) {
    Matrix<size_t> edgeMatrix = {{2, 1}, {0, 2}, {0, 0}};
    prior.setState(edgeMatrix);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    edgeMatrix = {{2, 1, 0}, {0, 2, 0}};
    prior.setState(edgeMatrix);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    expectConsistencyError = true;

}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_asymmetricEdgeMatrix_throwConsistencyError) {
    Matrix<size_t> edgeMatrix = {{2, 1}, {0, 2}};
    prior.setState(edgeMatrix);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    expectConsistencyError = true;
}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_incorrectEdgeNumber_throwConsistencyError) {
    Matrix<size_t> edgeMatrix = {{2, 1}, {1, 2}};
    prior.setState(edgeMatrix);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    expectConsistencyError = true;
}

class TestEdgeMatrixDeltaPrior: public ::testing::Test {
    public:
        MultiGraph graph = getUndirectedHouseMultiGraph();
        Matrix<size_t> edgeMatrix = {{10, 2}, {2, 10}};
        EdgeCountDeltaPrior edgeCountPrior = {7};
        BlockCountPoissonPrior blockCountPrior = {2};
        BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};
        EdgeMatrixDeltaPrior prior = {edgeMatrix, edgeCountPrior, blockPrior};

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

TEST_F(TestEdgeMatrixDeltaPrior, sample_returnSameEdgeMatrix){
    Matrix<size_t> edgeMatrix = prior.getState();
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

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihoodRatioFromBlockMove_forBlockMovePreservingEdgeMatrix_return0){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0)};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), 0);
}

TEST_F(TestEdgeMatrixDeltaPrior, getLogLikelihoodRatioFromBlockMove_forBlockMoveNotPreservingEdgeMatrix_returnMinusInfinity){
    BlockMove move = {0, blockPrior.getBlockOfIdx(0), blockPrior.getBlockOfIdx(0) + 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), -INFINITY);
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
    EXPECT_EQ(prior.getState().size(), prior.getBlockPrior().getBlockCount());

    auto sum = 0;
    for (auto er : prior.getState()){
        EXPECT_EQ(er.size(), prior.getBlockPrior().getBlockCount());
        for (auto ers : er){
            EXPECT_TRUE(ers >= 0);
            sum += ers;
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

TEST_F(TestEdgeMatrixUniformPrior, applyMove_forSomeBlockMove_changeEdgeMatrix){
    BlockMove move = {0, BLOCK_SEQUENCE[0], BLOCK_SEQUENCE[0] + 1};
    prior.applyBlockMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeBlockMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, BLOCK_SEQUENCE[0], BLOCK_SEQUENCE[0]+1};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromBlockMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyBlockMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);

}

TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_inconsistenBlockCount_ThrowConsistencyError){
    size_t originalBlockCount = blockCountPrior.getState();
    blockCountPrior.setState(10);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    blockCountPrior.setState(originalBlockCount);
}

TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_inconsistentEdgeCount_ThrowConsistencyError){
    size_t originalEdgeCount = edgeCountPrior.getState();
    edgeCountPrior.setState(50);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    edgeCountPrior.setState(originalEdgeCount);
}

}
