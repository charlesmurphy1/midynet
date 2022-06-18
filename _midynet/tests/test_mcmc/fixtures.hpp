#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{

class DummyRandomGraph: public RandomGraph{
    size_t m_edgeCount;

public:
    using RandomGraph::RandomGraph;
    DummyRandomGraph(size_t size): RandomGraph(size) {}

    void setGraph(const MultiGraph& graph) override{
        m_graph = graph;
        m_edgeCount = graph.getTotalEdgeNumber();
    }

    const size_t& getEdgeCount() const override { return m_edgeCount; }

    void sampleGraph() override { };
    virtual void samplePriors() override { };
    const double getLogLikelihood() const override { return 0; }
    const double getLogPrior() const override { return 0; }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override{ return 0; }
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0; }

};


}
