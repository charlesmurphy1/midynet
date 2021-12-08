#include "gtest/gtest.h"
#include <list>
#include <cmath>

#include "FastMIDyNet/dynamics/sis.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

const double INFECTION_PROB(0.7), RECOVERY_PROB(0.3), AUTO_INFECTION_PROB(1e-6);
const std::list<std::vector<int>> neighbor_states = {{1, 3}, {2, 2}, {3, 1}};


class TestSISDynamics: public::testing::Test{
public:
    FastMIDyNet::DummyRandomGraph graph = FastMIDyNet::DummyRandomGraph(7);
    FastMIDyNet::SISDynamics dynamics = FastMIDyNet::SISDynamics(graph, NUM_STEPS, INFECTION_PROB, RECOVERY_PROB, AUTO_INFECTION_PROB);
};


TEST_F(TestSISDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability) {

    for (auto neighbor_state: neighbor_states)
    EXPECT_EQ( (1-AUTO_INFECTION_PROB) * ( 1 - std::pow(1-INFECTION_PROB, neighbor_state[1])) + AUTO_INFECTION_PROB,
    dynamics.getActivationProb(neighbor_state));
}

TEST_F(TestSISDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {

    for (auto neighbor_state: neighbor_states)
    EXPECT_EQ(RECOVERY_PROB, dynamics.getDeactivationProb(neighbor_state));
}

}
