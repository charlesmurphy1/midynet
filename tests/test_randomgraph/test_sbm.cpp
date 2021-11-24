#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_BLOCKS = 3;
static const int NUM_EDGES = 100;
static const int NUM_VERTICES = 50;

class TestStochasticBlockModelFamily: public::testing::Test{
    public:
        BlockCountPoissonPrior blockCountPrior = {NUM_BLOCKS};
        BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};

        StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(blockPrior, edgeMatrixPrior);
        void SetUp() {
            randomGraph.sample();
        }
};

// void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge&, int, std::map<std::pair<BlockIndex, BlockIndex>, size_t>&);
// void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge&, int, std::map<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>, size_t>&);
// void getDiffEdgeMatMapFromBlockMove(const BlockMove&, std::map<std::pair<BlockIndex, BlockIndex>, size_t>&);
// double getLogLikelihoodRatio (const GraphMove&) ;
// double getLogLikelihoodRatio (const BlockMove&) ;
// double getLogPriorRatio (const GraphMove&) ;
// double getLogPriorRatio (const BlockMove&) ;
// double getLogJointRatio (const GraphMove& move) { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }
// double getLogJointRatio (const BlockMove& move) { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }
// void applyMove (const GraphMove&);
// void applyMove (const BlockMove&);
// void computationFinished(){
//     m_blockPrior.computationFinished();
//     m_edgeMatrixPrior.computationFinished();
// }
// static EdgeMatrix getEdgeMatrixFromGraph(const MultiGraph&, const BlockSequence&) ;
// static DegreeSequence getDegreeSequenceFromGraph(const MultiGraph&) ;
// static void checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat);
// static void checkGraphConsistencyWithDegreeSequence(const MultiGraph& graph, const DegreeSequence& degreeSeq) ;
// void checkSelfConsistency() ;

template<typename T>
void displayMatrix(Matrix<T> martrix, std::string name){
    std::cout << name << " = [" << std::endl;
    for (auto row : martrix){
        std::cout << "  [ ";
        for (auto col : row){
            std::cout << std::to_string(col) << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

TEST_F(TestStochasticBlockModelFamily, sampleState_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sampleState();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestStochasticBlockModelFamily, sample_graphChanges){
    for (size_t i = 0; i < 10; i++) {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihood_returnNonZeroValue){
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forAddedEdge){
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forRemovedEdge){
    auto neighbor = *randomGraph.getState().getNeighboursOfIdx(0).begin();
    BaseGraph::Edge removedEdge = {0, neighbor.vertexIndex};
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_F(TestStochasticBlockModelFamily, applyMove_forRemovedEdgeAndAddedEdge){
    auto neighbor = *randomGraph.getState().getNeighboursOfIdx(0).begin();
    BaseGraph::Edge removedEdge = {0, neighbor.vertexIndex};
    BaseGraph::Edge addedEdge = {0, 2};
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    FastMIDyNet::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(removedEdge);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicityIdx(addedEdge);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);

}

TEST_F(TestStochasticBlockModelFamily, applyMove_forNoEdgesAddedOrRemoved){
    FastMIDyNet::GraphMove move = {{}, {}};
    randomGraph.applyMove(move);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio){
    FastMIDyNet::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(TestStochasticBlockModelFamily, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio){

    auto neighbor = *randomGraph.getState().getNeighboursOfIdx(0).begin();
    BaseGraph::Edge removedEdge = {0, neighbor.vertexIndex};
    FastMIDyNet::GraphMove move = {{removedEdge}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatio(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}
