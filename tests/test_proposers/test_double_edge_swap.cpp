#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"
#include "fixtures.hpp"


class TestDoubleEdgeSwap: public ::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::DoubleEdgeSwap swapProposer;
        void SetUp() {
            swapProposer.setup(graph);
        }
};


TEST_F(TestDoubleEdgeSwap, SetUp_anyGraph_samplableSetContainsAllEdges) {
    EXPECT_EQ(graph.getTotalEdgeNumber(), swapProposer.getSamplableSet().total_weight());
    EXPECT_EQ(graph.getDistinctEdgeNumber(), swapProposer.getSamplableSet().size());
}
