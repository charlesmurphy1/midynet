#ifndef FAST_MIDYNET_PYTHON_VERTEXCOUNT_H
#define FAST_MIDYNET_PYTHON_VERTEXCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"


namespace FastMIDyNet{

template <typename VertexCountPriorBaseClass = VertexCountPrior>
class PyVertexCountPrior: public PyPrior<std::vector<size_t>, VertexCountPriorBaseClass> {
public:
    using PyPrior<std::vector<size_t>, VertexCountPriorBaseClass>::PyPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, VertexCountPriorBaseClass, getLogLikelihoodRatioFromBlockMove, move); }

    /* Overloaded abstract methods */
    void samplePriors() override { PYBIND11_OVERRIDE(void, VertexCountPriorBaseClass, samplePriors, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE(const double, VertexCountPriorBaseClass, getLogPrior, ); }
};

}

#endif
