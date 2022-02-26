#include "gtest/gtest.h"

#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/rng.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

class DummyEdgeProposer: public HingeFlipUniformProposer{
public:
    using HingeFlipUniformProposer::HingeFlipUniformProposer;
    const EdgeSampler& getEdgeSampler() { return m_edgeSampler; }
};

class TestHingeFlipUniformProposer: public::testing::Test {
    public:
        EdgeCountDeltaPrior edgeCountPrior = {3};
        ErdosRenyiFamily randomGraph = ErdosRenyiFamily(3, edgeCountPrior);
        DummyEdgeProposer proposer;
        MultiGraph graph;
        // BaseGraph::Edge existingEdge = {0, 2};
        void SetUp() {
            randomGraph.sample();
            graph = randomGraph.getGraph();

            // if (graph.getEdgeMultiplicityIdx(existingEdge)==0)
            //     graph.addEdgeIdx(existingEdge);
            // else
            //     graph.setEdgeMultiplicityIdx(existingEdge, 1);
            randomGraph.setGraph(graph);
            proposer.setUp(randomGraph);
        }
};


TEST_F(TestHingeFlipUniformProposer, setup_anyGraph_edgeSamplableSetContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), proposer.getEdgeSampler().getTotalWeight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), proposer.getEdgeSampler().getSize());
}


TEST_F(TestHingeFlipUniformProposer, setup_anyGraph_samplableSetHasOnlyOrderedEdges) {
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                EXPECT_EQ(round(proposer.getEdgeSampler().getEdgeWeight({vertex, neighbor.vertexIndex})), neighbor.label);
            else
                EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight({vertex, neighbor.vertexIndex}), 0);
}


TEST_F(TestHingeFlipUniformProposer, applyGraphMove_addExistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 2};
    GraphMove move = {{}, {edge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
}


TEST_F(TestHingeFlipUniformProposer, applyGraphMove_addMultiEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    GraphMove move = {{}, {edge, edge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+2);
}


TEST_F(TestHingeFlipUniformProposer, applyGraphMove_addInexistentEdge_edgeWeightIncreased) {
    BaseGraph::Edge edge = {0, 1};
    BaseGraph::Edge reversedEdge = {1, 0};
    GraphMove move = {{}, {edge}};
    proposer.applyGraphMove(move);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)+1);
    EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(reversedEdge), 0);
}


TEST_F(TestHingeFlipUniformProposer, applyGraphMove_removeEdge_edgeWeightDecreased) {
    auto edge = proposer.getEdgeSampler().sample();

    size_t edgeMult = graph.getEdgeMultiplicityIdx(edge);
    GraphMove move = {{edge}, {}};
    proposer.applyGraphMove(move);
    if (edgeMult > 1)
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicityIdx(edge)-1);
}

TEST_F(TestHingeFlipUniformProposer, getLogProposalProbRatio_forSomeGraphMove_returnCorrectValue) {

    for (size_t i=0; i<100; ++i){
        auto move = proposer.proposeMove();
        auto reversedMove = proposer.getReverseMove(move);
        double weight = proposer.getLogProposalWeight(move);
        proposer.applyGraphMove(move);
        double weightAfterMove = proposer.getLogProposalWeight(reversedMove);
        proposer.applyGraphMove(reversedMove);
        EXPECT_FLOAT_EQ(weightAfterMove - weight, proposer.getLogProposalProbRatio(move));
        proposer.applyGraphMove(move);
        randomGraph.applyGraphMove(move);
    }
}

}
