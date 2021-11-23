#include "gtest/gtest.h"

#include "FastMIDyNet/generators.h"


static const FastMIDyNet::Matrix<size_t> EDGE_MATRIX = {
    {4, 1, 3},
    {1, 0, 2},
    {3, 2, 6}
};
static const FastMIDyNet::BlockSequence VERTEX_BLOCKS = {
    0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2, 2, 2
};
static const FastMIDyNet::DegreeSequence DEGREES = {
    4, 2, 2, 0,
    2, 0, 1,
    0, 4, 4, 2, 1
};

static FastMIDyNet::Matrix<size_t> getEdgeMatrix(const FastMIDyNet::MultiGraph& graph, const std::vector<size_t>& vertexBlocks) {
    size_t blockNumber = 1;
    for (auto block: vertexBlocks)
        if (block >= blockNumber) blockNumber = block+1;

    FastMIDyNet::Matrix<size_t> edgeMatrix = {blockNumber, std::vector<size_t>(blockNumber, 0)};
    size_t block1, block2;

    for (auto vertex: graph) {
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (neighbor.vertexIndex > vertex)
                continue;

            block1 = vertexBlocks[vertex];
            block2 = vertexBlocks[neighbor.vertexIndex];
            edgeMatrix[block1][block2] += neighbor.label;
            edgeMatrix[block2][block1] += neighbor.label;
        }
    }
    return edgeMatrix;
}
static const size_t numberOfGeneratedGraphs = 10;

static const FastMIDyNet::DegreeSequence convertDegrees(const std::vector<size_t>& basegraphDegrees) {
    FastMIDyNet::DegreeSequence degrees;
    for (auto d: basegraphDegrees)
        degrees.push_back(d);
    return degrees;
}


TEST(TestDCSBMGenerator, generateDCSBM_givenEdgeMatrixAndDegrees_generatedGraphsRespectEdgeMatrixAndDegrees) {
    for (size_t i=0; i<numberOfGeneratedGraphs; i++) {
        auto randomGraph = FastMIDyNet::generateDCSBM(VERTEX_BLOCKS, EDGE_MATRIX, DEGREES);
        EXPECT_EQ(EDGE_MATRIX, getEdgeMatrix(randomGraph, VERTEX_BLOCKS));
        EXPECT_EQ(DEGREES, convertDegrees(randomGraph.getDegrees()));
    }
}

TEST(TestSBMGenerator, generate_SBM_givenEdgeMatrix_generatedGraphsRespectEdgeMatrix) {
    for (size_t i=0; i<numberOfGeneratedGraphs; i++) {
        auto randomGraph = FastMIDyNet::generateSBM(VERTEX_BLOCKS, EDGE_MATRIX);
        EXPECT_EQ(EDGE_MATRIX, getEdgeMatrix(randomGraph, VERTEX_BLOCKS));
    }
}
