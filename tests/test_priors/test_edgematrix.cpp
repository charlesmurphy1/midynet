#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/exceptions.h"
#include "fixtures.hpp"


const FastMIDyNet::BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0, 1, 1};


class DummyEdgeMatrixPrior: public FastMIDyNet::EdgeMatrixPrior {
    public:
        using EdgeMatrixPrior::EdgeMatrixPrior;
        void sampleState() {}
        double getLogLikelihood(const FastMIDyNet::Matrix<size_t>& state) const { return 0; }
        double getLogPrior() { return 0; }
        void checkSelfConsistency() const { }

        double getLogLikelihoodRatio(const FastMIDyNet::GraphMove&) const { return 0; }
        double getLogLikelihoodRatio(const FastMIDyNet::BlockMove&) const { return 0; }

        void applyMove(const FastMIDyNet::GraphMove&) { };
        void applyMove(const FastMIDyNet::BlockMove&) { };
};

class TestEdgeMatrixPrior: public ::testing::Test {
    public:
        FastMIDyNet::MultiGraph graph = getUndirectedHouseMultiGraph();
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = {2};
        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = {2};
        FastMIDyNet::BlockUniformPrior blockPrior = {graph.getSize(), blockCountPrior};

        DummyEdgeMatrixPrior edgeMatrixPrior = {edgeCountPrior, blockPrior};

        void SetUp() {
            blockPrior.setState(BLOCK_SEQUENCE);
            edgeCountPrior.setState(graph.getTotalEdgeNumber());
            edgeMatrixPrior.setGraph(graph);
        }
};


TEST_F(TestEdgeMatrixPrior, setGraph_anyGraph_edgeMatrixCorrectlySet) {
    EXPECT_EQ(edgeMatrixPrior.getState(),
            FastMIDyNet::Matrix<size_t>({{8, 6}, {6, 2}})
        );
    EXPECT_EQ(edgeMatrixPrior.getEdgesInBlock(), FastMIDyNet::BlockSequence({14, 8}));
}
