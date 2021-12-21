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
class PyEdgeCountPrior: public PyPrior<size_t, EdgeCountPriorBaseClass> {
public:
    using PyPrior<size_t, EdgeCountPriorBaseClass>::PyPrior;
    /* Pure abstract methods */
    double getLogLikelihoodFromState(const size_t& state) const override { PYBIND11_OVERRIDE_PURE(double, EdgeCountPriorBaseClass, getLogLikelihoodFromState, state); }

    /* Overloaded abstract methods */
    void samplePriors() override { PYBIND11_OVERRIDE(void, EdgeCountPriorBaseClass, samplePriors, ); }
    double getLogLikelihood() const override { PYBIND11_OVERRIDE(double, EdgeCountPriorBaseClass, getLogLikelihood, ); }
    double getLogPrior() override { PYBIND11_OVERRIDE(double, EdgeCountPriorBaseClass, getLogPrior, ); }
};

}

#endif
