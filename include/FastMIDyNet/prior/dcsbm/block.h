#ifndef FAST_MIDYNET_BLOCK_H
#define FAST_MIDYNET_BLOCK_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class BlockPrior: public Prior<BlockSequence>{
    size_t& m_size;
    BlockCountPrior& m_blockCountPrior;
public:
    BlockPrior(size_t size, BlockCountPrior& blockCountPrior, RNG& rng):
        m_size(size), m_blockCountPrior(blockCountPrior), Prior<BlockSequence>(rng) { }

    size_t getSize() { return m_size; }


};

}

#endif
