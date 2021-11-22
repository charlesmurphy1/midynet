#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <iostream>

#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_BLOCKS = 3;
static const int NUM_EDGES = 10;
static const int NUM_VERTICES = 7;

class TestStochasticBlockModelFamily: public::testing::Test{
    public:
        BlockCountPoissonPrior blockCountPrior = {NUM_BLOCKS};
        BlockUniformPrior blockPrior = {NUM_VERTICES, blockCountPrior};
        EdgeCountPoissonPrior edgeCountPrior = {NUM_EDGES};
        EdgeMatrixUniformPrior edgeMatrixPrior = EdgeMatrixUniformPrior(edgeCountPrior, blockPrior);

        StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(blockPrior, edgeMatrixPrior);
        void SetUp() {
            randomGraph.sample();
        }
};
