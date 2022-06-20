#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{

class DummySBM: public StochasticBlockModelFamily{
    size_t size = 10;
    size_t edgeCount = 25;
    size_t blockCount = 3;

    BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(blockCount);
    BlockUniformPrior blockPrior = BlockUniformPrior(size, blockCountPrior);
    EdgeCountDeltaPrior edgeCountPrior = EdgeCountDeltaPrior(edgeCount);
    EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);

public:
    DummySBM(): StochasticBlockModelFamily(size, blockPrior, edgeMatrixPrior) {}
};

class DummyER: public ErdosRenyiFamily{
    size_t size = 10;
    size_t edgeCount = 25;

    EdgeCountDeltaPrior edgeCountPrior = EdgeCountDeltaPrior(edgeCount);

public:
    DummyER(): ErdosRenyiFamily(size, edgeCountPrior) {}
};


}
