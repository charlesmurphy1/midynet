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

class DummyRandomGraph: public RandomGraph{
    std::vector<size_t> m_blocks;
    size_t m_blockCount = 1;
    std::vector<size_t> m_vertexCounts;
    Matrix<size_t> m_edgeMatrix;
    std::vector<size_t> m_edgeCounts;
    size_t m_edgeCount;
    std::vector<size_t> m_degrees;
    std::vector<CounterMap<size_t>> m_degreeCounts;
public:
    using RandomGraph::RandomGraph;
    DummyRandomGraph(size_t size): RandomGraph(size), m_blocks(size, 0), m_vertexCounts(1, size){}

    void setGraph(const MultiGraph& graph) override{
        m_graph = graph;
        m_edgeCount = graph.getTotalEdgeNumber();
        m_edgeMatrix = Matrix<size_t>(1, {1, 2 * m_edgeCount});
        m_edgeCounts = std::vector<size_t>(1, 2 * m_edgeCount);
        m_degreeCounts = computeDegreeCountsInBlocks();
        m_degrees = graph.getDegrees();
    }

    const std::vector<BlockIndex>& getBlocks() const override { return m_blocks; }
    const size_t& getBlockCount() const override { return m_blockCount; }
    const std::vector<size_t>& getVertexCountsInBlocks() const override { return m_vertexCounts; }
    const Matrix<size_t>& getEdgeMatrix() const override { return m_edgeMatrix; }
    const std::vector<size_t>& getEdgeCountsInBlocks() const override { return m_edgeCounts; }
    const size_t& getEdgeCount() const override { return m_edgeCount; }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const override { return m_degreeCounts; }
    const std::vector<size_t>& getDegrees() const override { return m_degrees; }

    void sampleGraph() override { };
    virtual void samplePriors() override { };
    double getLogLikelihood() const override { return 0; }
    double getLogPrior() const override { return 0; }
    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override{ return 0; }
    double getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0; }
    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override { return 0; }
    double getLogPriorRatioFromBlockMove(const BlockMove& move) const override { return 0; }

};


}
