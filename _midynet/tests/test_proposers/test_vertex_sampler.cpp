#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/vertex_sampler.h"


namespace FastMIDyNet{

class TestVertexUniformSampler: public ::testing::Test{
public:
    VertexUniformSampler sampler = VertexUniformSampler();
    MultiGraph graph = MultiGraph(10);
    void SetUp(){
        sampler.setUp(graph);
    }
};

TEST_F(TestVertexUniformSampler, setUp_withGraph){
    sampler.setUp(graph);
    EXPECT_EQ(sampler.getTotalWeight(), 10);
}

TEST_F(TestVertexUniformSampler, sample_returnVertexInGraph){
    sampler.setUp(graph);
    for (size_t i=0; i<100; ++i){
        auto vertex = sampler.sample();
    }
}

TEST_F(TestVertexUniformSampler, update_forRemovedEdge_doNothing){
    graph.addEdgeIdx(0, 1);
    graph.addEdgeIdx(0, 2);
    graph.addEdgeIdx(0, 3);
    GraphMove move = {{{0, 1}}, {}};
    sampler.setUp(graph);

    EXPECT_EQ(sampler.getTotalWeight(), 10);
    sampler.update(move);
    EXPECT_EQ(sampler.getTotalWeight(), 10);
}

TEST_F(TestVertexUniformSampler, update_forAddedEdge_doNothing){
    graph.addEdgeIdx(0, 1);
    GraphMove move = {{}, {{0, 2}}};
    sampler.setUp(graph);

    EXPECT_EQ(sampler.getTotalWeight(), 10);
    sampler.update(move);
    EXPECT_EQ(sampler.getTotalWeight(), 10);
}

TEST_F(TestVertexUniformSampler, getTotalWeight_returnSizeOfVertexSet){
    MultiGraph otherGraph = MultiGraph(7);
    sampler.setUp(otherGraph);
    EXPECT_EQ(sampler.getTotalWeight(), 7);

}

class TestVertexDegreeSampler: public ::testing::Test{
public:
    double shift = 3;
    size_t vertexCount = 5;
    VertexDegreeSampler sampler = VertexDegreeSampler(shift);
    MultiGraph graph = MultiGraph(vertexCount);
    std::vector<size_t> degrees;
    size_t edgeCount;

    void SetUp(){
        graph.addEdgeIdx(0, 1);
        graph.addEdgeIdx(0, 2);
        graph.addEdgeIdx(0, 3);

        graph.addEdgeIdx(1, 1);
        graph.addEdgeIdx(1, 2);
        graph.addEdgeIdx(1, 3);
        degrees = graph.getDegrees();
        edgeCount = graph.getTotalEdgeNumber();

        sampler.setUp(graph);
    }
};

TEST_F(TestVertexDegreeSampler, setUp_withGraph){
    for (auto vertex : graph){
        EXPECT_GT(sampler.getVertexWeight(vertex), 0);
    }
}

TEST_F(TestVertexDegreeSampler, update_forRemovedEdge_changeWeight){
    GraphMove move = {{{0,1}}, {}};

    sampler.update(move);
    EXPECT_EQ(sampler.getVertexWeight(0), shift + degrees[0] - 1);
    EXPECT_EQ(sampler.getVertexWeight(1), shift + degrees[1] - 1);


    move = {{{1,1}}, {}};
    sampler.update(move);
    EXPECT_EQ(sampler.getVertexWeight(1), shift + degrees[1] - 3);
}

TEST_F(TestVertexDegreeSampler, update_forAddedEdge_changeWeight){
    GraphMove move = {{}, {{2,3}}};

    sampler.update(move);
    EXPECT_EQ(sampler.getVertexWeight(2), shift + degrees[2] + 1);
    EXPECT_EQ(sampler.getVertexWeight(3), shift + degrees[3] + 1);
}

TEST_F(TestVertexDegreeSampler, getTotalWeight_returnCorrectWeight){
    EXPECT_EQ(sampler.getTotalWeight(), shift * vertexCount + 2 * edgeCount);
}

}
