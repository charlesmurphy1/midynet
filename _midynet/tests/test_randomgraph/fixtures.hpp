#ifndef FASTMIDYNET_DYNAMICS_FIXTURES_HPP
#define FASTMIDYNET_DYNAMICS_FIXTURES_HPP

#include <iostream>
#include "gtest/gtest.h"

#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/sbm.h"
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

class DummyGraphLikelihood: public GraphLikelihoodModel{
public:
    const double getLogLikelihood() const { return 0; }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const { return 0; }
};

class DummyRandomGraph: public RandomGraph{
    size_t m_edgeCount;
    DummyGraphLikelihood likelihood;
    void setUpLikelihood() override { };
public:
    using RandomGraph::RandomGraph;
    DummyRandomGraph(size_t size): RandomGraph(size, likelihood) { }

    void setGraph(const MultiGraph graph) override{
        RandomGraph::setGraph(graph);
        m_edgeCount = graph.getTotalEdgeNumber();
    }

    const size_t& getEdgeCount() const override { return m_edgeCount; }

    void sampleState() override { };
};



} // namespace FastMIDyNet


#endif
