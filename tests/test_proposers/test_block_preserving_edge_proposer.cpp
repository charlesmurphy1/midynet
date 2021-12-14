#include "gtest/gtest.h"

#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/proposer/edge_proposer/block_edge_proposer.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

size_t NUM_BLOCKS = 3;
size_t NUM_VERTICES = 100;
size_t NUM_EDGES = 250;
class TestBlockPreservingHingeFlip: public::testing::Test {
public:
    BlockCountDeltaPrior blockCountPrior = {NUM_BLOCKS};
    BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
    EdgeCountDeltaPrior edgeCountPrior = {NUM_EDGES};
    EdgeMatrixUniformPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(blockPrior, edgeMatrixPrior);


    BlockPreservingEdgeProposer<HingeFlip> proposer = BlockPreservingEdgeProposer<HingeFlip>();
    void SetUp() {
        randomGraph.sample();
        proposer.setBlocks( randomGraph.getBlocks() );
        proposer.setUp(randomGraph);
    }

    void TearDown(){

    }
};

TEST_F(TestBlockPreservingHingeFlip, proposeMove_returnMovePreservingBlocks){
    auto move = proposer.proposeMove();
}


}


// TEST_F(TestHingeFlip, setup_anyGraph_edgeSamplableSetContainsAllEdges) {
//     EXPECT_EQ(graph.getTotalEdgeNumber(), flipProposer.getEdgeSamplableSet().total_weight());
//     EXPECT_EQ(graph.getDistinctEdgeNumber(), flipProposer.getEdgeSamplableSet().size());
// }
//
//
// TEST_F(TestHingeFlip, setup_anyGraph_nodeSamplableSetContainsAllEdges) {
//     EXPECT_EQ(graph.getSize(), flipProposer.getNodeSamplableSet().size());
// }
//
//
// TEST_F(TestHingeFlip, setup_anyGraph_samplableSetHasOnlyOrderedEdges) {
//     for (auto vertex: graph)
//         for (auto neighbor: graph.getNeighboursOfIdx(vertex))
//             if (vertex <= neighbor.vertexIndex)
//                 EXPECT_EQ(round(flipProposer.getEdgeSamplableSet().get_weight({vertex, neighbor.vertexIndex})), neighbor.label);
//             else
//                 EXPECT_EQ(flipProposer.getEdgeSamplableSet().count({vertex, neighbor.vertexIndex}), 0);
// }
//
//
// TEST_F(TestHingeFlip, updateProbabilities_addExistentEdge_edgeWeightIncreased) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{}, {edge}};
//     flipProposer.updateProbabilities(move);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
// }
//
//
// TEST_F(TestHingeFlip, updateProbabilities_addMultiEdge_edgeWeightIncreased) {
//     BaseGraph::Edge edge = {0, 1};
//     FastMIDyNet::GraphMove move = {{}, {edge, edge}};
//     flipProposer.updateProbabilities(move);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
// }
//
//
// TEST_F(TestHingeFlip, updateProbabilities_addInexistentEdge_edgeWeightIncreased) {
//     BaseGraph::Edge edge = {0, 1};
//     BaseGraph::Edge reversedEdge = {1, 0};
//     FastMIDyNet::GraphMove move = {{}, {edge}};
//     flipProposer.updateProbabilities(move);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().count(reversedEdge), 0);
// }
//
//
// TEST_F(TestHingeFlip, updateProbabilities_removeEdge_edgeWeightDecreased) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{edge}, {}};
//     flipProposer.updateProbabilities(move);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
// }
//
//
// TEST_F(TestHingeFlip, updateProbabilities_removeMultiEdge_edgeWeightDecreased) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{edge, edge}, {}};
//     flipProposer.updateProbabilities(move);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().get_weight(edge), graph.getEdgeMultiplicityIdx(edge)-2);
// }
//
//
// TEST_F(TestHingeFlip, updateProbabilities_removeAllEdges_edgeRemovedFromSamplableSet) {
//     BaseGraph::Edge edge = {0, 2};
//     FastMIDyNet::GraphMove move = {{edge, edge, edge}, {}};
//     flipProposer.updateProbabilities(move);
//     EXPECT_EQ(flipProposer.getEdgeSamplableSet().count(edge), 0);
// }
