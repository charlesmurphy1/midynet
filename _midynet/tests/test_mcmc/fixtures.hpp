#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/proposer/block_proposer/uniform.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{

class DummyRandomGraph{
public:
    size_t GRAPH_SIZE = 100;
    size_t BLOCK_COUNT = 5;
    size_t EDGE_COUNT = 250;
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    UniformBlockProposer blockProposer = UniformBlockProposer(0.);
    BlockCountDeltaPrior blockCount = BlockCountDeltaPrior(BLOCK_COUNT);
    BlockUniformPrior blockPrior = BlockUniformPrior(GRAPH_SIZE, blockCount);
    EdgeCountDeltaPrior edgeCount = EdgeCountDeltaPrior(EDGE_COUNT);
    EdgeMatrixUniformPrior edgeMatrix = EdgeMatrixUniformPrior(edgeCount, blockPrior);
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(GRAPH_SIZE, blockPrior, edgeMatrix);
};

}
