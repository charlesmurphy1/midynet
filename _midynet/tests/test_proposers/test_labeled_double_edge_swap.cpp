#include "gtest/gtest.h"

#include "fixtures.hpp"
#include "FastMIDyNet/proposer/edge_proposer/labeled_double_edge_swap.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

    class DummyLabeledDoubleEdgeSwapProposer: public LabeledDoubleEdgeSwapProposer{
    public:
        LabelPairSampler getLabelSampler() { return m_labelSampler;}
        std::unordered_map<LabelPair, EdgeSampler*> getEdgeSamplers(){ return m_labeledEdgeSampler; }
    };

class TestLabeledDoubleEdgeSwapProposer: public ::testing::Test{
public:
    DummySBMGraph randomGraph = DummySBMGraph();
    DummyLabeledDoubleEdgeSwapProposer proposer = DummyLabeledDoubleEdgeSwapProposer();
    void SetUp(){
        randomGraph.sample();
        proposer.setUp(randomGraph);
    }
};

TEST_F(TestLabeledDoubleEdgeSwapProposer, proposeMove){
    auto move = proposer.proposeMove();
    for (auto edge : move.removedEdges)
        EXPECT_GT(randomGraph.getGraph().getEdgeMultiplicityIdx(edge), 0);

    for (auto edge : move.addedEdges)
        EXPECT_GE(randomGraph.getGraph().getEdgeMultiplicityIdx(edge), 0);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, onLabelCreation_doNothing){
    BlockMove move = {0, 0, 0};
    proposer.onLabelCreation(move);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, onLabelDeletion_doNothing){
    BlockMove move = {0, 0, 0};
    proposer.onLabelDeletion(move);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, getLogProposalProbRatio_return0){
    auto move = proposer.proposeMove();
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyGraphMove_forEdgeAdded){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    proposer.applyGraphMove({{}, {{0,1}}});
    EXPECT_EQ(totalEdgeCountBefore + 1, proposer.getTotalEdgeCount());

}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyGraphMove_forSelfLoopAdded){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    proposer.applyGraphMove({{}, {{0,0}}});
    EXPECT_EQ(totalEdgeCountBefore + 1, proposer.getTotalEdgeCount());
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyGraphMove_forSomeGraphMove){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    auto move = proposer.proposeMove();
    proposer.applyGraphMove(move);
    EXPECT_EQ(totalEdgeCountBefore, proposer.getTotalEdgeCount());
}

TEST_F(TestLabeledDoubleEdgeSwapProposer, applyBlockMove_forSomeBlockMove){
    for (auto vertex : randomGraph.getGraph()){
        BlockIndex prevBlockIdx = randomGraph.getBlockOfIdx(vertex);
        BlockIndex nextBlockIdx = (prevBlockIdx == randomGraph.getBlockCount()-1) ? prevBlockIdx - 1 : prevBlockIdx + 1;
        BlockMove move = {vertex, prevBlockIdx, nextBlockIdx};
        // EXPECT_TRUE(proposer.getVertexSamplers().at(prevBlockIdx)->contains(vertex));
        // for (auto neighbor : randomGraph.getGraph().getNeighboursOfIdx(vertex)){
        //     auto edge = getOrderedEdge({vertex, neighbor.vertexIndex});
        //     auto labelPair = proposer.getLabelSampler().getLabelOfIdx(edge);
        //     EXPECT_TRUE(proposer.getEdgeSamplers().at(labelPair)->contains(edge));
        // }
        // double totalWeightBefore = proposer.getVertexSamplers().at(prevBlockIdx)->getTotalWeight();

        proposer.applyBlockMove(move);

        // EXPECT_FALSE(proposer.getVertexSamplers().at(prevBlockIdx)->contains(vertex));
        double expectedWeightDiff = 0;
        for (auto neighbor : randomGraph.getGraph().getNeighboursOfIdx(vertex)){
            auto s = proposer.getLabelSampler().getLabelOfIdx(neighbor.vertexIndex);
            auto edge = getOrderedEdge({vertex, neighbor.vertexIndex});
            auto prevLabelPair = getOrderedPair<BlockIndex>({prevBlockIdx, s});
            auto nextLabelPair = getOrderedPair<BlockIndex>({nextBlockIdx, s});
            // EXPECT_FALSE(proposer.getEdgeSamplers().at(prevLabelPair)->contains(edge));
            // EXPECT_TRUE(proposer.getEdgeSamplers().at(nextLabelPair)->contains(edge));
            // if (s != prevBlockIdx){
            //     expectedWeightDiff += neighbor.label;
            // }
        }
        // EXPECT_EQ(proposer.getVertexSamplers().at(prevBlockIdx)->getTotalWeight(), totalWeightBefore - 1);
        randomGraph.applyBlockMove(move);
    }
}


}
