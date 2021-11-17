#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/random_graph/dcsbm.h"

namespace FastMIDyNet{

class DegreePrior: public Prior<DegreeSequence>{
public:
    DegreePrior();

};

}

#endif
