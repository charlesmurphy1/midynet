#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/ising-glauber.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

static const double COUPLING_CONSTANT = 2.;
static const std::list<std::vector<int>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}};

static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}

class TestIsingGlauber: public::testing::Test{
public:
    EdgeCountDeltaPrior edgeCountPrior = {10};
    ErdosRenyiFamily graph = ErdosRenyiFamily(10, edgeCountPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    FastMIDyNet::IsingGlauberDynamics dynamics = FastMIDyNet::IsingGlauberDynamics(graph, NUM_STEPS, COUPLING_CONSTANT, false);
};


TEST_F(TestIsingGlauber, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighborState: NEIGHBOR_STATES){
        EXPECT_EQ(
            sigmoid( 2 * COUPLING_CONSTANT * (neighborState[0]-neighborState[1]) ),
            dynamics.getActivationProb(neighborState)
        );
    }
}

TEST_F(TestIsingGlauber, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighborState: NEIGHBOR_STATES){
        EXPECT_EQ(sigmoid(
            -2*COUPLING_CONSTANT*(neighborState[0]-neighborState[1])),
            dynamics.getDeactivationProb(neighborState)
        );
    }
}

TEST_F(TestIsingGlauber, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sample();
    edgeProposer.setUp(graph);
    auto graphMove = edgeProposer.proposeMove();
    double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(graphMove);
    double logLikelihoodAfter = dynamics.getLogLikelihood();

    EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

}
