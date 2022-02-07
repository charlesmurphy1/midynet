#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_sampler.h"


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
        sampler.setUp(graph);
    }
};

TEST_F(TestEdgeSampler, setUp_withGraph){
    sampler.setUp(graph);
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
    sampler.removeEdge({0, 1});
    EXPECT_EQ(sampler.getEdgeWeight({0, 1}), 0);
    EXPECT_EQ(sampler.getTotalWeight(), edgeCount - 1);
}

TEST_F(TestEdgeSampler, addEdge_addEdgeToSampler){
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 0);
    sampler.addEdge({2, 3});
    EXPECT_EQ(sampler.getEdgeWeight({2, 3}), 1);
    EXPECT_EQ(sampler.getTotalWeight(), edgeCount + 1);
}



}
