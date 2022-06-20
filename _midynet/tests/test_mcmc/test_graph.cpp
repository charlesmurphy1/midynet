#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "fixtures.hpp"
#include "FastMIDyNet/mcmc/callbacks/collector.h"
#include "FastMIDyNet/mcmc/graph.hpp"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/rng.h"


using namespace std;

namespace FastMIDyNet{

class TestGraphMCMC: public::testing::Test{
public:

    DummyER randomGraph = DummyER();
    GraphMCMC mcmc = GraphMCMC(randomGraph);

    CollectLikelihoodOnSweep likelihoodCollector = CollectLikelihoodOnSweep();
    CollectEdgeMultiplicityOnSweep edgeCollector = CollectEdgeMultiplicityOnSweep();

    bool expectConsistencyError = false;
    void SetUp(){
        randomGraph.sample();

        mcmc.insertCallBack("likelihood", likelihoodCollector);
        mcmc.insertCallBack("edge", edgeCollector);

        mcmc.setUp();
        mcmc.checkSafety();

    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
        mcmc.tearDown();
    }
};

TEST_F(TestGraphMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestGraphMCMC, doMHSweep){
    mcmc.doMHSweep(1000);
}

class TestVertexLabeledGraphMCMC: public::testing::Test{
public:

    DummySBM randomGraph = DummySBM();
    BlockUniformProposer proposer = BlockUniformProposer();
    VertexLabeledGraphMCMC<BlockIndex> mcmc = VertexLabeledGraphMCMC<BlockIndex>(randomGraph, proposer);

    CollectLikelihoodOnSweep likelihoodCollector = CollectLikelihoodOnSweep();
    CollectEdgeMultiplicityOnSweep edgeCollector = CollectEdgeMultiplicityOnSweep();
    CollectPartitionOnSweep partitionCollector = CollectPartitionOnSweep();

    bool expectConsistencyError = false;
    void SetUp(){
        randomGraph.sample();

        mcmc.insertCallBack("likelihood", likelihoodCollector);
        mcmc.insertCallBack("edge", edgeCollector);
        mcmc.insertCallBack("partition", partitionCollector);

        mcmc.setUp();
        mcmc.checkSafety();

    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
        mcmc.tearDown();
    }
};

TEST_F(TestVertexLabeledGraphMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestVertexLabeledGraphMCMC, doMHSweep){
    mcmc.doMHSweep(1000);
}


} // FastMIDyNet
