#include "gtest/gtest.h"

#include "fixtures.hpp"
#include "FastMIDyNet/proposer/edge_proposer/labeled_hinge_flip.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class DummyLabeledHingeFlipUniformProposer: public LabeledHingeFlipUniformProposer{
public:
    LabelPairSampler getLabelSampler() { return m_labelSampler;}
    std::unordered_map<LabelPair, EdgeSampler*> getEdgeSamplers(){ return m_labeledEdgeSampler; }
    std::unordered_map<BlockIndex, VertexSampler*> getVertexSamplers(){ return m_labeledVertexSampler; }
};

class TestLabeledHingeFlipUniformProposer: public ::testing::Test{
public:
    DummySBMGraph randomGraph = DummySBMGraph();
    DummyLabeledHingeFlipUniformProposer proposer = DummyLabeledHingeFlipUniformProposer();
    void SetUp(){
        randomGraph.sample();
        proposer.setUp(randomGraph);
        proposer.checkSafety();
    }
    void TearDown() {
        proposer.checkConsistency();
    }
};

TEST_F(TestLabeledHingeFlipUniformProposer, proposeMove){
    auto move = proposer.proposeMove();
    LabelPair rs = {10, 10};
    for (auto edge : move.removedEdges){
        EXPECT_GT(randomGraph.getGraph().getEdgeMultiplicityIdx(edge), 0);
        if (rs.first == 10 and rs.second == 10)
            rs = proposer.getLabelSampler().getLabelOfIdx(edge);
        EXPECT_EQ(rs, proposer.getLabelSampler().getLabelOfIdx(edge));
    }
    for (auto edge : move.addedEdges){
        EXPECT_GE(randomGraph.getGraph().getEdgeMultiplicityIdx(edge), 0);
        EXPECT_EQ(rs, proposer.getLabelSampler().getLabelOfIdx(edge));
    }
}

TEST_F(TestLabeledHingeFlipUniformProposer, onLabelCreation_doNothing){
    BlockMove move = {0, 0, 0};
    proposer.onLabelCreation(move);
}

TEST_F(TestLabeledHingeFlipUniformProposer, onLabelDeletion_doNothing){
    BlockMove move = {0, 0, 0};
    proposer.onLabelDeletion(move);
}

TEST_F(TestLabeledHingeFlipUniformProposer, getLogProposalProbRatio_return0){
    auto move = proposer.proposeMove();
    EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestLabeledHingeFlipUniformProposer, applyGraphMove_forEdgeAdded){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    proposer.applyGraphMove({{}, {{0,1}}});
    EXPECT_EQ(totalEdgeCountBefore + 1, proposer.getTotalEdgeCount());

}

TEST_F(TestLabeledHingeFlipUniformProposer, applyGraphMove_forSelfLoopAdded){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    proposer.applyGraphMove({{}, {{0,0}}});
    EXPECT_EQ(totalEdgeCountBefore + 1, proposer.getTotalEdgeCount());
}

TEST_F(TestLabeledHingeFlipUniformProposer, applyGraphMove_forSomeGraphMove){
    size_t totalEdgeCountBefore = proposer.getTotalEdgeCount();
    auto move = proposer.proposeMove();
    auto removedEdge = move.removedEdges[0];
    auto rs = proposer.getLabelSampler().getLabelOfIdx(removedEdge);
    proposer.applyGraphMove(move);
    EXPECT_EQ(totalEdgeCountBefore, proposer.getTotalEdgeCount());
}

TEST_F(TestLabeledHingeFlipUniformProposer, applyBlockMove_forSomeBlockMove){
    for (auto vertex : randomGraph.getGraph()){
        BlockIndex prevBlockIdx = randomGraph.getBlockOfIdx(vertex);
        BlockIndex nextBlockIdx = (prevBlockIdx == randomGraph.getBlockCount()-1) ? prevBlockIdx - 1 : prevBlockIdx + 1;
        BlockMove move = {vertex, prevBlockIdx, nextBlockIdx};
        proposer.applyBlockMove(move);
        randomGraph.applyBlockMove(move);
    }
}


}
