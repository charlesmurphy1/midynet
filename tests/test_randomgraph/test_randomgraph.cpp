#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <iostream>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_VERTICES = 7;
static const GraphMove GRAPH_MOVE = {{{0, 3}}, {{0, 5}}};
static MultiGraph GRAPH = getUndirectedHouseMultiGraph();

class TestRandomGraphBaseClass: public::testing::Test{
    public:
        DummyRandomGraph randomGraph = DummyRandomGraph(NUM_VERTICES);
        MultiGraph graph = GRAPH;


        void SetUp() {
            randomGraph.setState(graph);
        }
};


// void enumerateAllGraphs() const;

TEST_F(TestRandomGraphBaseClass, getState_returnHouseMultigraphGraph){
    MultiGraph graph = randomGraph.getState();
    EXPECT_EQ(graph.getSize(), NUM_VERTICES);
    EXPECT_EQ(graph.getTotalEdgeNumber(), 10);
}

TEST_F(TestRandomGraphBaseClass, setState_differentFromHouseGraph){
    MultiGraph graph = randomGraph.getState();
    graph.addEdgeIdx(0, 0);
    randomGraph.setState(graph);
    EXPECT_EQ(graph.getSize(), NUM_VERTICES);
    EXPECT_EQ(graph.getTotalEdgeNumber(), 11);
}

TEST_F(TestRandomGraphBaseClass, getSize_returnCorrectGraphSize){
    MultiGraph graph = randomGraph.getState();
    EXPECT_EQ(graph.getSize(), randomGraph.getSize());
    EXPECT_EQ(NUM_VERTICES, randomGraph.getSize());
}

TEST_F(TestRandomGraphBaseClass, getLogJoint_return0){
    EXPECT_EQ(randomGraph.getLogJoint(), 0);
}

TEST_F(TestRandomGraphBaseClass, applyMove_forSomeGraphMove_changeGraph){
    randomGraph.applyMove(GRAPH_MOVE);
    EXPECT_FALSE( randomGraph.getState().isEdgeIdx(GRAPH_MOVE.removedEdges[0]) );
    EXPECT_TRUE( randomGraph.getState().isEdgeIdx(GRAPH_MOVE.addedEdges[0]) );
}
