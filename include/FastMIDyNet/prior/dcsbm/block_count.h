#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/random_graph/dcsbm.h"

namespace FastMIDyNet{

class BlockCountPrior: public Prior<size_t>{

public:
    BlockCountPrior();
};

}


#endif
