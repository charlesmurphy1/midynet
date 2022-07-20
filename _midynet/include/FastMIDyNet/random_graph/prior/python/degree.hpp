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
class PyDegreePrior: public PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, DegreePriorBaseClass> {
public:
    using PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, DegreePriorBaseClass>::PyVertexLabeledPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogLikelihoodRatioFromGraphMove, move);
    }
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogLikelihoodRatioFromLabelMove, move);
    }

    /* Overloaded abstract methods */
    void setState(const DegreeSequence& state) override {
        PYBIND11_OVERRIDE(void, DegreePriorBaseClass, setState, state);
    }
    const DegreeCountsMap& getDegreeCounts() const override {
        PYBIND11_OVERRIDE(const DegreeCountsMap&, DegreePriorBaseClass, getDegreeCounts, );
    }
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogPriorRatioFromGraphMove, move);
    }
    const double getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogPriorRatioFromLabelMove, move);
    }
    void checkSelfConsistency() const override {
        PYBIND11_OVERRIDE_PURE(void, DegreePriorBaseClass, checkSelfConsistency, );
    }
};

}

#endif
