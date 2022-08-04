#ifndef FASTMIDYNET_GRAPH_FIXTURES_HPP
#define FASTMIDYNET_GRAPH_FIXTURES_HPP

#include "FastMIDyNet/types.h"

#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/hsbm.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"

#include "FastMIDyNet/dynamics/sis.hpp"

#include "FastMIDyNet/mcmc/mcmc.h"


namespace FastMIDyNet{

static MultiGraph getUndirectedHouseMultiGraph(){
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
    const MultiGraph sample() const { return generateErdosRenyi(*m_sizePtr, *m_edgeCountPtr); }
    const double getLogLikelihood() const { return 0; }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const { return 0; }
    size_t* m_sizePtr = nullptr;
    size_t* m_edgeCountPtr = nullptr;
};


class DummyRandomGraph: public RandomGraph{
    size_t m_edgeCount;
    DummyGraphLikelihood likelihood;
    void setUpLikelihood() override { likelihood.m_sizePtr = &m_size; likelihood.m_edgeCountPtr = &m_edgeCount; }
public:
    using RandomGraph::RandomGraph;
    DummyRandomGraph(size_t size): RandomGraph(size, likelihood) { }

    void setState(const MultiGraph state) override{
        RandomGraph::setState(state);
        m_edgeCount = state.getTotalEdgeNumber();
    }

    const size_t getEdgeCount() const override { return m_edgeCount; }
};

class DummyErdosRenyiGraph: public ErdosRenyiModelBase{
public:
    EdgeCountDeltaPrior prior;
    DummyErdosRenyiGraph(size_t size=10, size_t edgeCount=25):
    ErdosRenyiModelBase(size), prior(edgeCount) { setEdgeCountPrior(prior); }
};

class DummySBMGraph: public StochasticBlockModelBase{
    size_t size;
    size_t edgeCount;
    size_t blockCount;

    BlockCountDeltaPrior blockCountPrior;
    BlockUniformPrior blockPrior;
    EdgeCountDeltaPrior edgeCountPrior;
    LabelGraphErdosRenyiPrior labelGraphPrior;

public:
    DummySBMGraph(size_t size=10, size_t edgeCount=25, size_t blockCount=3):
    StochasticBlockModelBase(size),
    blockCountPrior(blockCount),
    blockPrior(size, blockCountPrior),
    edgeCountPrior(edgeCount),
    labelGraphPrior(edgeCountPrior, blockPrior)
     {
        setLabelGraphPrior(labelGraphPrior);
    }
    using StochasticBlockModelBase::sample;
};

class DummyRestrictedSBMGraph: public StochasticBlockModelBase{
    size_t size;
    size_t edgeCount;
    size_t blockCount;

    BlockCountDeltaPrior blockCountPrior;
    BlockUniformHyperPrior blockPrior;
    EdgeCountDeltaPrior edgeCountPrior;
    LabelGraphErdosRenyiPrior labelGraphPrior;

public:
    DummyRestrictedSBMGraph(size_t size=10, size_t edgeCount=25, size_t blockCount=3):
    StochasticBlockModelBase(size),
    blockCountPrior(blockCount),
    blockPrior(size, blockCountPrior),
    edgeCountPrior(edgeCount),
    labelGraphPrior(edgeCountPrior, blockPrior)
     {
        setLabelGraphPrior(labelGraphPrior);
    }
    using StochasticBlockModelBase::sample;
};

class DummyNestedSBMGraph: public NestedStochasticBlockModelBase{
    size_t size;
    size_t edgeCount;
    size_t blockCount;

    EdgeCountDeltaPrior edgeCountPrior;
    LabelGraphErdosRenyiPrior labelGraphPrior;

public:
    DummyNestedSBMGraph(size_t size=10, size_t edgeCount=25):
    NestedStochasticBlockModelBase(size),
    edgeCountPrior(edgeCount) { setEdgeCountPrior(edgeCountPrior); }
};

class DummyDynamics: public Dynamics<RandomGraph>{
public:
    DummyDynamics(RandomGraph& graphPrior, size_t numStates=2, double numSteps = 10):
    Dynamics<RandomGraph>(graphPrior, numStates, numSteps){}

    const double getTransitionProb(
        VertexState prevVertexState,
        VertexState nextVertexState,
        VertexNeighborhoodState vertexNeighborhoodState
    ) const { return 1. / getNumStates(); }

    void updateNeighborsStateFromEdgeMove(
        BaseGraph::Edge edge,
        int direction,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&prev,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence>&next
    ) const { Dynamics::updateNeighborsStateFromEdgeMove(edge, direction, prev, next); }
};

class DummySISDynamics: public SISDynamics<RandomGraph>{
public:
    DummySISDynamics(RandomGraph& graphPrior, size_t numSteps=10, double infection = 0.1):
    SISDynamics<RandomGraph>(graphPrior, numSteps, infection){}
};

class DummyLabeledSISDynamics: public SISDynamics<VertexLabeledRandomGraph<BlockIndex>>{
public:
    DummyLabeledSISDynamics(VertexLabeledRandomGraph<BlockIndex>& graphPrior, size_t numSteps=10, double infection = 0.1):
    SISDynamics<VertexLabeledRandomGraph<BlockIndex>>(graphPrior, numSteps, infection){}
};


class DummyMCMC: public MCMC{
public:
    bool doMetropolisHastingsStep() override {
        onStepBegin();
        m_lastLogJointRatio = 0;
        m_lastLogAcceptance = -log(2);
        if (m_uniform(rng) < exp(m_lastLogAcceptance))
            m_isLastAccepted = true;
        else
            m_isLastAccepted = false;
            onStepEnd();
        return m_isLastAccepted;

    }
    void sample() override { }
    void samplePrior() override { }
    const double getLogLikelihood() const override { return 1; }
    const double getLogPrior() const override { return 2; }
    const double getLogJoint() const override { return getLogLikelihood() + getLogPrior(); }
};





}
#endif
