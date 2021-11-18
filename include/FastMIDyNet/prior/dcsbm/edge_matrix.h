#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class EdgeMatrixPrior: public Prior< Matrix<size_t> >{
    EdgeCountPrior& m_edgeCountPrior;
public:
    EdgeMatrixPrior(EdgeCountPrior& edgeCountPrior): m_edgeCountPrior(edgeCountPrior) { };

    const size_t& getEdgeCount() { return m_edgeCountPrior.getState(); }

};

}

#endif
