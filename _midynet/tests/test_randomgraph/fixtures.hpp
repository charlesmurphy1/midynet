#ifndef FASTMIDYNET_DYNAMICS_FIXTURES_HPP
#define FASTMIDYNET_DYNAMICS_FIXTURES_HPP

#include <iostream>
#include "gtest/gtest.h"

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"
#include "BaseGraph/undirected_multigraph.h"



namespace FastMIDyNet{

static FastMIDyNet::MultiGraph getUndirectedHouseMultiGraph(){
    //     /*
    //      * (0)      (1)
    //      * ||| \   / | \
    //      * |||  \ /  |  \
    //      * |||   X   |  (4)
    //      * |||  / \  |  /
    //      * ||| /   \ | /
    //      * (2)------(3)-----(5)
    //      *
    //      *      (6)
    //      */
    // STATE = {0,0,0,1,1,1,2}
    // NEIGHBORS_STATE = {{3, 1, 0}, {1, 2, 0}, {4, 1, 0}, {3, 1, 1}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}}
    FastMIDyNet::MultiGraph graph(7);
    graph.addMultiedgeIdx(0, 2, 3);
    graph.addEdgeIdx(0, 3);
    graph.addEdgeIdx(1, 2);
    graph.addEdgeIdx(1, 3);
    graph.addEdgeIdx(1, 4);
    graph.addEdgeIdx(2, 3);
    graph.addEdgeIdx(3, 4);
    graph.addEdgeIdx(3, 5);

    return graph;

}

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


} // namespace FastMIDyNet


#endif
