#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/random_graph/dcsbm.h"

namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t>{
public:
    EdgeCountPrior();
};

}

#endif
