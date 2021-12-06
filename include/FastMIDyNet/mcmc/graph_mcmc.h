#ifndef FAST_MIDYNET_GRAPH_MCMC_H
#define FAST_MIDYNET_GRAPH_MCMC_H

#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/blockproposer/blockproposer.h"
#include "FastMIDyNet/proposer/movetypes.h"

namespace FastMIDyNet{

class RandomGraphMCMC{
protected:
    RandomGraph& m_randomGraph;
public:
    RandomGraphMCMC(RandomGraph& randomGraph): m_randomGraph(randomGraph) {}
    virtual void setUp() { };
    virtual void doMetropolisHastingsStep(double beta) { };

};

class StochasticBlockGraphMCMC: public RandomGraphMCMC{
private:
    BlockProposer& m_blockProposer;
public:
    StochasticBlockGraphMCMC(StochasticBlockModelFamily& randomGraph, BlockProposer& blockProposer):
    RandomGraphMCMC(randomGraph), m_blockProposer(blockProposer) {}

    void setUp() {
        m_blockProposer.setUp(m_randomGraph.getBlockSequence(), m_randomGraph.getBlockCount());
    };
    void doMetropolisHastingsStep(double beta) {
        BlockMove move = m_blockProposer.proposeMove();
    };

};

}

#endif
