#ifndef FAST_MIDYNET_PYTHON_EDGEMATRIX_H
#define FAST_MIDYNET_PYTHON_EDGEMATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"


namespace FastMIDyNet{

template <typename BaseClass = EdgeMatrixPrior>
class PyEdgeMatrixPrior: public PyPrior<std::vector<std::vector<size_t>>, BaseClass> {
public:
    using PyPrior<std::vector<std::vector<size_t>>, BaseClass>::PyPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromBlockMove, move); }

    /* Overloaded abstract methods */
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPriorRatioFromGraphMove, move); }
    const double getLogPriorRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPriorRatioFromBlockMove, move); }
};

}

#endif
