#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class EdgeMatrixPrior: public Prior< Matrix<size_t> >{

public:
    EdgeMatrixPrior();
};

}

#endif
