#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "fixtures.hpp"


const FastMIDyNet::BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};


class DummyEdgeMatrixPrior: public FastMIDyNet::EdgeMatrixPrior {
    public:
        using EdgeMatrixPrior::EdgeMatrixPrior;
        void sampleState() {}
        double getLogLikelihood(const FastMIDyNet::Matrix<size_t>& state) const { return 0; }

        double getLogLikelihoodRatio(const FastMIDyNet::GraphMove&) const { return 0; }
        double getLogLikelihoodRatio(const FastMIDyNet::BlockMove&) const { return 0; }

        void applyMove(const FastMIDyNet::GraphMove&) { };
        void applyMove(const FastMIDyNet::BlockMove&) { };


        void _createBlock() { createBlock(); }
        void _destroyBlock(const FastMIDyNet::BlockIndex& block) { destroyBlock(block); }
        void _moveEdgesInBlocks(const FastMIDyNet::BlockMove& move) { moveEdgesInBlocks(move); }
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
    EXPECT_EQ(edgeMatrixPrior.getEdgesInBlock(), FastMIDyNet::BlockSequence({14, 8}));
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
    EXPECT_EQ(edgeMatrixPrior.getEdgesInBlock(), std::vector<size_t>({14, 8, 0}));
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
        EXPECT_EQ(edgeMatrixPrior.getEdgesInBlock(), std::vector<size_t>({19, 3}));

        removedBlock++;
    }
}

TEST_F(TestEdgeMatrixPrior, moveEdgesInBlocks_vertexChangingBlock_neighborsOfVertexChangedOfBlocks) {
    edgeMatrixPrior._moveEdgesInBlocks({0, 0, 1});
    EXPECT_EQ(edgeMatrixPrior.getState(),
            FastMIDyNet::Matrix<size_t>({{6, 4}, {4, 8}})
        );
}

TEST_F(TestEdgeMatrixPrior, moveEdgesInBlocks_vertexChangingBlockWithSelfloop_neighborsOfVertexChangedOfBlocks) {
    edgeMatrixPrior._moveEdgesInBlocks({5, 1, 0});
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
