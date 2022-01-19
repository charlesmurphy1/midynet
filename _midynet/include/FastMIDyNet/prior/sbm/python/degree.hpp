#ifndef FAST_MIDYNET_PYTHON_DEGREE_H
#define FAST_MIDYNET_PYTHON_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/degree.h"


namespace FastMIDyNet{

template <typename DegreePriorBaseClass = DegreePrior>
class PyDegreePrior: public PyPrior<std::vector<size_t>, DegreePriorBaseClass> {
public:
    using PyPrior<std::vector<size_t>, DegreePriorBaseClass>::PyPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogLikelihoodRatioFromGraphMove, move);
    }
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogLikelihoodRatioFromBlockMove, move);
    }

    /* Overloaded abstract methods */
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const override {
        PYBIND11_OVERRIDE(const std::vector<CounterMap<size_t>>&, DegreePriorBaseClass, getDegreeCountsInBlocks, );
    }
};

}

#endif
