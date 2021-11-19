#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <iostream>

#include "FastMIDyNet/prior/block.h"
#include "FastMIDyNet/prior/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/types.h"
#include "fixtures.hpp"

using namespace std;
using namespace FastMIDyNet;

static const int NUM_VERTICES = 7;

class TestStochasticBlockModelFamily: public::testing::Test{
    public:

        StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(NUM_VERTICES);
        void SetUp() {
            randomGraph.sample();
        }
};
