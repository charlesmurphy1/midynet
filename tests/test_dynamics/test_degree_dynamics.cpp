#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/degree_dynamics.h"
#include "fixtures.hpp"


static const double C = 1.;
static const std::list<std::vector<int>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}, {2, 0}};


class TestDegreeDynamics: public::testing::Test{
public:
    FastMIDyNet::RNG rng;
    FastMIDyNet::DummyRandomGraph graph = FastMIDyNet::DummyRandomGraph(7, rng);
    FastMIDyNet::DegreeDynamics dynamic = FastMIDyNet::DegreeDynamics(graph, rng, C);
};


TEST_F(TestDegreeDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ((neighbor_state[0] + neighbor_state[1])/C,
                  dynamic.getActivationProb(neighbor_state));
}

TEST_F(TestDegreeDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(1 - (neighbor_state[0] + neighbor_state[1])/C,
                  dynamic.getDeactivationProb(neighbor_state));
}
