#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"


namespace FastMIDyNet {


class BlockProposer: public Proposer<BlockMove> { };

} // namespace FastMIDyNet


#endif
