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
static const GraphMove GRAPH_MOVE = {{{0, 2}}, {{0, 5}}};
static MultiGraph GRAPH = getUndirectedHouseMultiGraph();

class TestRandomGraphBaseClass: public::testing::Test{
    public:
        DummyRandomGraph randomGraph = DummyRandomGraph(NUM_VERTICES);
        MultiGraph graph = GRAPH;
        void SetUp() {
            randomGraph.setState(graph);
        }
};


// const MultiGraph& getState() const { return m_state; }
// void setState(const MultiGraph& state) { m_state = state; }
// const int getSize() { return m_size; }
// void copyState(const MultiGraph& state);
//
// void sample() {
//     samplePriors();
//     sampleState();
// };
// virtual void sampleState() = 0;
// virtual void samplePriors() = 0;
// double getLogLikelihood() const { return 0.; };
// double getLogPrior() const { return 0.; }
// double getLogJoint() const { return getLogLikelihood() + getLogPrior(); }
//
// virtual double getLogJointRatio (const GraphMove& move) const = 0;
// void applyMove(const GraphMove& move);
// void enumerateAllGraphs() const;
// void doMetropolisHastingsStep(double beta=1.0) { };
// void checkConsistency() { };

TEST_F(TestRandomGraphBaseClass, getState_returnHouseMultigraphGraph){
    
}
