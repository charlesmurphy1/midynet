#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/sbm.h"


namespace FastMIDyNet {

class BlockProposer: public Proposer<BlockMove> {
public:
    virtual void setUp(const StochasticBlockModelFamily& sbmGraph) = 0;
};

} // namespace FastMIDyNet


#endif
