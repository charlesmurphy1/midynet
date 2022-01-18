#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <iostream>

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

namespace FastMIDyNet {

const int NUM_VERTICES = 7;
const int NUM_STATES = 3;
const vector<int> ALL_VERTEX_STATES = {0, 1, 2};
const State STATE = {0, 0, 0, 1, 1, 2, 0};
const NeighborsState NEIGHBORS_STATE = {
    {3, 1, 0}, {1, 2, 0}, {4, 1, 0}, {3, 1, 1}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}
};
const GraphMove GRAPH_MOVE = {{{0, 2}}, {{0, 5}}};
MultiGraph GRAPH = getUndirectedHouseMultiGraph();

class TestDynamicsBaseClass: public::testing::Test{
    public:
        DummyRandomGraph randomGraph = DummyRandomGraph(NUM_VERTICES);
        DummyDynamics dynamics = DummyDynamics(randomGraph, NUM_STATES, NUM_STEPS);
        MultiGraph graph = GRAPH;
        State state = STATE;
        void SetUp() {
            dynamics.setState(state);
            dynamics.setGraph(graph);
        }
};

TEST_F(TestDynamicsBaseClass, getState_returnCurrentState){
    auto x = dynamics.getState();
    EXPECT_EQ(x, STATE);
}

TEST_F(TestDynamicsBaseClass, getGraph_returnCurrentGraph){
    auto g = dynamics.getGraph();
    EXPECT_EQ(g, GRAPH);
}

TEST_F(TestDynamicsBaseClass, getSize_returnGraphSize){
    auto n = dynamics.getSize();
    EXPECT_EQ(n, STATE.size());
    EXPECT_EQ(n, NUM_VERTICES);
}

TEST_F(TestDynamicsBaseClass, getNumStates_returnNumStates){
    auto s = dynamics.getNumStates();
    EXPECT_EQ(s, NUM_STATES);
}

TEST_F(TestDynamicsBaseClass, getRandomState_returnRandomState){
    auto x = dynamics.getRandomState();
    EXPECT_EQ(x.size(), NUM_VERTICES);
    for (auto xx : x){
        EXPECT_TRUE(xx <= NUM_STATES - 1);
        EXPECT_TRUE(xx >= 0);

    }
}

TEST_F(TestDynamicsBaseClass, getNeighborsState_forSomeState_returnThatNeighborState){
    auto neighborState = dynamics.getNeighborsState(STATE);
    EXPECT_EQ(neighborState.size(), NUM_VERTICES);
    int i = 0, j = 0;
    for (auto vertexNeighborState: neighborState){
        j = 0;
        EXPECT_EQ(vertexNeighborState.size(), NUM_STATES);
        for (auto l : vertexNeighborState){
            EXPECT_EQ(l, NEIGHBORS_STATE[i][j]);
            j++;
        }
        i++;
    }
}

TEST_F(TestDynamicsBaseClass, getTransitionProbs_forEachVertexState_returnTransitionProbVector){
    vector<double> probs;
    for (auto in_state : ALL_VERTEX_STATES){
        probs = dynamics.getTransitionProbs(in_state, NEIGHBORS_STATE[3]);
        EXPECT_EQ(probs.size(), NUM_STATES);
        for (auto out_state : ALL_VERTEX_STATES){
            EXPECT_EQ(probs[ out_state ], dynamics.getTransitionProb(in_state, out_state, NEIGHBORS_STATE[3]));
        }
    }
}

TEST_F(TestDynamicsBaseClass, sampleState_forSomeNumSteps_returnNothing){
    dynamics.sampleState();
}

TEST_F(TestDynamicsBaseClass, getPastStates_returnPastStates){
    dynamics.sampleState();
    StateSequence past_states = dynamics.getPastStates();
    EXPECT_EQ(past_states.size(), NUM_STEPS) ;
    for (auto state : past_states){
        EXPECT_EQ(state.size(), NUM_VERTICES) ;
    }
}

TEST_F(TestDynamicsBaseClass, getFutureStates_returnFutureStates){
    dynamics.sampleState();
    StateSequence future_states = dynamics.getFutureStates();
    EXPECT_EQ(future_states.size(), NUM_STEPS) ;
    for (auto state : future_states){
        EXPECT_EQ(state.size(), NUM_VERTICES) ;
    }
}

TEST_F(TestDynamicsBaseClass, getLogJointRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sampleState();
    double ratio = dynamics.getLogJointRatioFromGraphMove(GRAPH_MOVE);
    EXPECT_EQ(ratio, 0.);
}

TEST_F(TestDynamicsBaseClass, applyMove_forSomeGraphMove_expectChangesInTheGraph){
    dynamics.sampleState();
    dynamics.applyGraphMove(GRAPH_MOVE);
    auto graph = dynamics.getGraph();
    EXPECT_EQ(graph.getEdgeMultiplicityIdx(0, 2), 2);
    EXPECT_EQ(graph.getEdgeMultiplicityIdx(0, 5), 1);
}


} /* FastMIDyNet */
