#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/wilson_cowan.h"
#include "fixtures.hpp"


static const double A = 1., NU = 7., MU = 1., ETA = 0.5;
static const std::list<std::vector<int>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}, {2, 0}};

static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}

class TestWilsonCowan: public::testing::Test{
    public:
        FastMIDyNet::RNG rng;
        FastMIDyNet::DummyRandomGraph graph = FastMIDyNet::DummyRandomGraph(7, rng);
        FastMIDyNet::WilsonCowanDynamics dynamic = FastMIDyNet::WilsonCowanDynamics(graph, rng, A, NU, MU, ETA);
};


TEST_F(TestWilsonCowan, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(sigmoid(A*(NU*neighbor_state[1] - MU)),
                  dynamic.getActivationProb(neighbor_state));
}

TEST_F(TestWilsonCowan, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(ETA,
                  dynamic.getDeactivationProb(neighbor_state));
}
