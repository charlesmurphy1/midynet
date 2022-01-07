#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "fixtures.hpp"


const FastMIDyNet::BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};


class DummyEdgeMatrixPrior: public FastMIDyNet::EdgeMatrixPrior {
    public:
        using EdgeMatrixPrior::EdgeMatrixPrior;
        void sampleState() {}
        double getLogLikelihood() const { return 0.; }

        double getLogLikelihoodRatioFromGraphMove(const FastMIDyNet::GraphMove&) const { return 0; }
        double getLogLikelihoodRatioFromBlockMove(const FastMIDyNet::BlockMove&) const { return 0; }

        void applyGraphMove(const FastMIDyNet::GraphMove&) { };
        void applyBlockMove(const FastMIDyNet::BlockMove&) { };


        void _createBlock() { createBlock(); }
        void _destroyBlock(const FastMIDyNet::BlockIndex& block) { destroyBlock(block); }
        void _moveEdgeCountsInBlocks(const FastMIDyNet::BlockMove& move) { moveEdgeCountsInBlocks(move); }
};

class TestEdgeMatrixPrior: public ::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = {2};
        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = {2};
        FastMIDyNet::BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};

        DummyEdgeMatrixPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};

        void SetUp() {
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());
            edgeMatrixPrior.setGraph(graph);
        }
};


TEST_F(TestEdgeMatrixPrior, setGraph_anyGraph_edgeMatrixCorrectlySet) {
    EXPECT_EQ(edgeMatrixPrior.getState(),
            FastMIDyNet::Matrix<size_t>({{8, 6}, {6, 2}})
        );
    EXPECT_EQ(edgeMatrixPrior.getEdgeCountsInBlocks(), FastMIDyNet::BlockSequence({14, 8}));
}

TEST_F(TestEdgeMatrixPrior, samplePriors_anyGraph_returnSumOfPriors) {
    double tmp = edgeMatrixPrior.getLogPrior();
    edgeMatrixPrior.computationFinished();
    EXPECT_EQ(tmp, edgeCountPrior.getLogJoint()+blockPrior.getLogJoint());
}

TEST_F(TestEdgeMatrixPrior, createBlock_anySetup_addRowAndColumnToEdgeMatrix) {
    edgeMatrixPrior._createBlock();
    EXPECT_EQ(edgeMatrixPrior.getState(),
            FastMIDyNet::Matrix<size_t>( {{8, 6, 0}, {6, 2, 0}, {0, 0, 0}} ));
}

TEST_F(TestEdgeMatrixPrior, createBlock_anySetup_addElementToEdgeCountOfBlocks) {
    edgeMatrixPrior._createBlock();
    EXPECT_EQ(edgeMatrixPrior.getEdgeCountsInBlocks(), std::vector<size_t>({14, 8, 0}));
}

TEST_F(TestEdgeMatrixPrior, destroyBlock_anyBlock_removeFirstRowAndColumn) {
    size_t removedBlock = 0;

    for (const FastMIDyNet::BlockSequence& blockSequence:
            std::list<FastMIDyNet::BlockSequence>{{1, 1, 1, 1, 1, 2, 1}, {0, 0, 0, 0, 0, 2, 0}, {0, 0, 0, 0, 0, 1, 0}}) {

        blockPrior.setState(blockSequence);
        blockCountPrior.setState(3); // blockPrior setState changes blockCountPrior state
        edgeMatrixPrior.setGraph(graph);

        edgeMatrixPrior._destroyBlock(removedBlock);
        EXPECT_EQ(edgeMatrixPrior.getState(),
                FastMIDyNet::Matrix<size_t>( {{18, 1}, {1, 2}} ));
        EXPECT_EQ(edgeMatrixPrior.getEdgeCountsInBlocks(), std::vector<size_t>({19, 3}));

        removedBlock++;
    }
}

TEST_F(TestEdgeMatrixPrior, moveEdgesInBlocks_vertexChangingBlock_neighborsOfVertexChangedOfBlocks) {
    edgeMatrixPrior._moveEdgeCountsInBlocks({0, 0, 1});
    EXPECT_EQ(edgeMatrixPrior.getState(),
            FastMIDyNet::Matrix<size_t>({{6, 4}, {4, 8}})
        );
}

TEST_F(TestEdgeMatrixPrior, moveEdgesInBlocks_vertexChangingBlockWithSelfloop_neighborsOfVertexChangedOfBlocks) {
    edgeMatrixPrior._moveEdgeCountsInBlocks({5, 1, 0});
    EXPECT_EQ(edgeMatrixPrior.getState(),
            FastMIDyNet::Matrix<size_t>({{12, 5}, {5, 0}})
        );
}
TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_validData_noThrow) {
    EXPECT_NO_THROW(edgeMatrixPrior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_incorrectBlockNumber_throwConsistencyError) {
    blockCountPrior.setState(1);
    EXPECT_THROW(edgeMatrixPrior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
    blockCountPrior.setState(3);
    EXPECT_THROW(edgeMatrixPrior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_edgeMatrixNotOfBlockNumberSize_throwConsistencyError) {
    FastMIDyNet::Matrix<size_t> edgeMatrix = {{2, 1}, {0, 2}, {0, 0}};
    edgeMatrixPrior.setState(edgeMatrix);
    EXPECT_THROW(edgeMatrixPrior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
    edgeMatrix = {{2, 1, 0}, {0, 2, 0}};
    edgeMatrixPrior.setState(edgeMatrix);
    EXPECT_THROW(edgeMatrixPrior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_asymmetricEdgeMatrix_throwConsistencyError) {
    FastMIDyNet::Matrix<size_t> edgeMatrix = {{2, 1}, {0, 2}};
    edgeMatrixPrior.setState(edgeMatrix);
    EXPECT_THROW(edgeMatrixPrior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}

TEST_F(TestEdgeMatrixPrior, checkSelfConsistency_incorrectEdgeNumber_throwConsistencyError) {
    FastMIDyNet::Matrix<size_t> edgeMatrix = {{2, 1}, {1, 2}};
    edgeMatrixPrior.setState(edgeMatrix);
    EXPECT_THROW(edgeMatrixPrior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}

class TestEdgeMatrixUniformPrior: public ::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = {2};
        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = {2};
        FastMIDyNet::BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};

        FastMIDyNet::EdgeMatrixUniformPrior prior = {edgeCountPrior, blockPrior};

        void SetUp() {
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());
            prior.setGraph(graph);
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
    double expectedLogLikelihood = -FastMIDyNet::logMultisetCoefficient( B * (B + 1) / 2, E);

    EXPECT_EQ(actualLogLikelihood, expectedLogLikelihood);
}

TEST_F(TestEdgeMatrixUniformPrior, applyMove_forSomeGraphMove_changeEdgeMatrix){
    FastMIDyNet::GraphMove move = {{{0, 0}}, {{0, 2}}};
    prior.applyGraphMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeGraphMoveContainingASelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{{0, 0}}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeGraphMoveChangingTheEdgeCount_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBeforeMove = prior.getLogLikelihood();
    prior.applyGraphMove(move);
    double logLikelihoodAfterMove = prior.getLogLikelihood();
    double expectedLogLikelihood = logLikelihoodAfterMove - logLikelihoodBeforeMove ;

    EXPECT_EQ(actualLogLikelihoodRatio, expectedLogLikelihood);
}

TEST_F(TestEdgeMatrixUniformPrior, applyMove_forSomeBlockMove_changeEdgeMatrix){
    FastMIDyNet::BlockMove move = {0, BLOCK_SEQUENCE[0], BLOCK_SEQUENCE[0] + 1};
    prior.applyBlockMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeMatrixUniformPrior, getLogLikelihoodRatio_forSomeBlockMove_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::BlockMove move = {0, BLOCK_SEQUENCE[0], BLOCK_SEQUENCE[0]+1};
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
    blockCountPrior.setState(10);
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}

TEST_F(TestEdgeMatrixUniformPrior, checkSelfConsistency_inconsistentEdgeCount_ThrowConsistencyError){
    edgeCountPrior.setState(50);
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}
