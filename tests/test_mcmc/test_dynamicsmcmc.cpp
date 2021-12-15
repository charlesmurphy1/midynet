#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/proposer/block_proposer/uniform_proposer.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"

#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/dynamics/sis.h"
#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/rng.h"

size_t GRAPH_SIZE = 100;
size_t BLOCK_COUNT = 3;
size_t EDGE_COUNT = 250;
size_t NUM_STEPS = 100;

using namespace std;

namespace FastMIDyNet{

class TestDynamicsMCMC: public::testing::Test{
public:
    UniformBlockProposer blockProposer = UniformBlockProposer(0.);
    BlockCountDeltaPrior blockCount = BlockCountDeltaPrior(BLOCK_COUNT);
    VertexCountUniformPrior vertexCount = VertexCountUniformPrior(GRAPH_SIZE, blockCount);
    BlockHyperPrior blocks = BlockHyperPrior(vertexCount);
    EdgeCountDeltaPrior edgeCount = EdgeCountDeltaPrior(EDGE_COUNT);
    EdgeMatrixUniformPrior edgeMatrix = EdgeMatrixUniformPrior(edgeCount, blocks);
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(blocks, edgeMatrix);
    StochasticBlockGraphMCMC graphmcmc = StochasticBlockGraphMCMC(randomGraph, blockProposer);

    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    SISDynamics dynamics = SISDynamics(randomGraph, NUM_STEPS, 0.5);
    DynamicsMCMC mcmc = DynamicsMCMC(dynamics, graphmcmc, edgeProposer, 1., 1., 0.);
    void SetUp(){
        setSeed(time(NULL));
        mcmc.sample();
        mcmc.setUp();

    }
    void TearDown(){
        mcmc.tearDown();
    }
};

TEST_F(TestDynamicsMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}


} // FastMIDyNet
