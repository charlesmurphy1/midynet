#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"


namespace FastMIDyNet{

class TestVertexUniformSampler: public ::testing::Test{
public:
    VertexUniformSampler sampler = VertexUniformSampler();
    MultiGraph graph = MultiGraph(10);
    void setUpSamplerWithGraph(const MultiGraph& graph){
        sampler.clear();
        for (auto vertex : graph)
            sampler.onVertexInsertion(vertex);
        for (auto vertex : graph)
            for (auto neighbor : graph.getNeighboursOfIdx(vertex))
                if (vertex <= neighbor.vertexIndex)
                    sampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);

    }
    void SetUp(){
        setUpSamplerWithGraph(graph);
    }
};

TEST_F(TestVertexUniformSampler, setUp_withGraph){
    setUpSamplerWithGraph(graph);
    EXPECT_EQ(sampler.getTotalWeight(), 10);
}

TEST_F(TestVertexUniformSampler, sample_returnVertexInGraph){
    setUpSamplerWithGraph(graph);
    for (size_t i=0; i<100; ++i){
        auto vertex = sampler.sample();
    }
}

TEST_F(TestVertexUniformSampler, removeEdge_doNothing){
    graph.addEdgeIdx(0, 1);
    graph.addEdgeIdx(0, 2);
    graph.addEdgeIdx(0, 3);
    setUpSamplerWithGraph(graph);
    EXPECT_EQ(sampler.getTotalWeight(), 10);
    sampler.onEdgeRemoval({0, 1});
    EXPECT_EQ(sampler.getTotalWeight(), 10);
}

TEST_F(TestVertexUniformSampler, addEdge_doNothing){
    graph.addEdgeIdx(0, 1);
    setUpSamplerWithGraph(graph);

    EXPECT_EQ(sampler.getTotalWeight(), 10);
    sampler.onEdgeAddition({0, 2});
    EXPECT_EQ(sampler.getTotalWeight(), 10);
}

TEST_F(TestVertexUniformSampler, getTotalWeight_returnSizeOfVertexSet){
    MultiGraph otherGraph = MultiGraph(7);
    setUpSamplerWithGraph(otherGraph);
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


    void setUpSamplerWithGraph(const MultiGraph& graph){
        sampler.clear();
        for (auto vertex : graph)
            sampler.onVertexInsertion(vertex);
        for (auto vertex : graph)
            for (auto neighbor : graph.getNeighboursOfIdx(vertex))
                if (vertex <= neighbor.vertexIndex)
                    sampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
    }

    void SetUp(){
        graph.addEdgeIdx(0, 1);
        graph.addEdgeIdx(0, 2);
        graph.addEdgeIdx(0, 3);

        graph.addEdgeIdx(1, 1);
        graph.addEdgeIdx(1, 2);
        graph.addEdgeIdx(1, 3);
        degrees = graph.getDegrees();
        edgeCount = graph.getTotalEdgeNumber();

        setUpSamplerWithGraph(graph);
    }
};

TEST_F(TestVertexDegreeSampler, setUp_withGraph){
    for (auto vertex : graph){
        EXPECT_GT(sampler.getVertexWeight(vertex), 0);
    }
}

TEST_F(TestVertexDegreeSampler, removeEdge_changeWeight){
    EXPECT_EQ(sampler.getVertexWeight(0), shift + degrees[0]);
    EXPECT_EQ(sampler.getVertexWeight(1), shift + degrees[1]);
    sampler.onEdgeRemoval({0, 1});
    EXPECT_EQ(sampler.getVertexWeight(0), shift + degrees[0] - 1);
    EXPECT_EQ(sampler.getVertexWeight(1), shift + degrees[1] - 1);

    sampler.onEdgeRemoval({1, 1});
    EXPECT_EQ(sampler.getVertexWeight(1), shift + degrees[1] - 3);
}

TEST_F(TestVertexDegreeSampler, eraseVertex_changeWeightAndDoesNotContainVertex){
    EXPECT_TRUE(sampler.contains(1));
    EXPECT_EQ(sampler.getTotalWeight(), shift * vertexCount + edgeCount);
    sampler.onVertexErasure(1);
    EXPECT_FALSE(sampler.contains(1));
    EXPECT_EQ(sampler.getTotalWeight(), shift * (vertexCount - 1) + edgeCount);
    EXPECT_EQ(sampler.getVertexWeight(1), 0);
}

TEST_F(TestVertexDegreeSampler, addEdge_changeWeight){
    sampler.onEdgeAddition({2, 3});
    EXPECT_EQ(sampler.getVertexWeight(2), shift + degrees[2] + 1);
    EXPECT_EQ(sampler.getVertexWeight(3), shift + degrees[3] + 1);
}

TEST_F(TestVertexDegreeSampler, getTotalWeight_returnCorrectWeight){
    EXPECT_EQ(sampler.getTotalWeight(), shift * vertexCount + edgeCount);
}

}
