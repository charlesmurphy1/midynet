#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"


namespace FastMIDyNet {


class BlockProposer: public Proposer<BlockMove> {
public:
    virtual void setUp(StochasticBlockModelFamily& sbmGraph) { };
};
} // namespace FastMIDyNet


#endif
