#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"


namespace FastMIDyNet{

class TestEdgeSampler: public ::testing::Test{
public:
    size_t vertexCount = 5;
    EdgeSampler sampler = EdgeSampler();
    MultiGraph graph = MultiGraph(vertexCount);
    size_t edgeCount;

    void SetUp(){
        graph.addEdgeIdx(0, 1);
        graph.addEdgeIdx(0, 2);
        graph.addEdgeIdx(0, 3);

        graph.addEdgeIdx(1, 1);
        graph.addEdgeIdx(1, 2);
        graph.addEdgeIdx(1, 3);

        edgeCount = graph.getTotalEdgeNumber();
        sampler.setUpWithGraph(graph);
    }
};

TEST_F(TestEdgeSampler, setUpwithGraph){
    EXPECT_EQ(sampler.getTotalWeight(), edgeCount);
}

TEST_F(TestEdgeSampler, getEdgeWeight_returnCorrectWeight){
    EXPECT_EQ(sampler.getEdgeWeight({0, 1}), 1);
    EXPECT_EQ(sampler.getEdgeWeight({1, 1}), 1);
}

TEST_F(TestEdgeSampler, sample_returnEdgeInGraph){
    for(size_t i=0; i<100; ++i){
        auto edge = sampler.sample();
        EXPECT_GT(graph.getEdgeMultiplicityIdx(edge), 0);
    }
}

TEST_F(TestEdgeSampler, removeEdge_removeEdgeFromSampler){
    GraphMove move = {{{0, 1}}, {}};
    sampler.onEdgeRemoval({0, 1});
    EXPECT_EQ(sampler.getEdgeWeight({0, 1}), 0);
    EXPECT_EQ(sampler.getTotalWeight(), edgeCount - 1);
}

TEST_F(TestEdgeSampler, addEdge_addEdgeToSampler){
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 0);
    sampler.onEdgeAddition({2, 3});
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 1);
    EXPECT_EQ(sampler.getTotalWeight(), edgeCount + 1);
}

TEST_F(TestEdgeSampler, onEdgeInsertion_forEdgeWeightBiggerThanMaxWeight_edgeWeightIsCutOff){
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 0);
    sampler.onEdgeInsertion({2, 3}, 105);
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
    // EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
}

TEST_F(TestEdgeSampler, onEdgeAddition_forEdgeWeightBiggerThanMaxWeight_edgeWeightRemainsTheSame){
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 0);
    sampler.onEdgeInsertion({2, 3}, 105);
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
    sampler.onEdgeAddition({2, 3});
    sampler.onEdgeAddition({3, 2});
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
}


TEST_F(TestEdgeSampler, onEdgeRemoval_forEdgeWeightBiggerThanMaxWeight_edgeWeightRemainsTheSame){
    graph.addMultiedgeIdx(2, 3, 105);
    sampler.setUpWithGraph(graph);
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
    sampler.onEdgeRemoval({2, 3});
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
}
TEST_F(TestEdgeSampler, onEdgeRemoval_forEdgeWeightEqualToMaxWeight_edgeWeightDecrements){
    graph.addMultiedgeIdx(2, 3, 100);
    sampler.setUpWithGraph(graph);
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 100);
    sampler.onEdgeRemoval({2, 3});
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 99);
}



}
