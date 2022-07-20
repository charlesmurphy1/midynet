#ifndef FAST_MIDYNET_PYTHON_EDGECOUNT_H
#define FAST_MIDYNET_PYTHON_EDGECOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"


namespace FastMIDyNet{

template <typename EdgeCountPriorBaseClass = EdgeCountPrior>
class PyEdgeCountPrior: public PyVertexLabeledPrior<size_t, BlockIndex, EdgeCountPriorBaseClass> {
public:
    using PyVertexLabeledPrior<size_t, BlockIndex, EdgeCountPriorBaseClass>::PyVertexLabeledPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodFromState(const size_t& state) const override { PYBIND11_OVERRIDE_PURE(const double, EdgeCountPriorBaseClass, getLogLikelihoodFromState, state); }

    /* Overloaded abstract methods */
};

}

#endif
