#ifndef FAST_MIDYNET_PYTHON_LABELED_DEGREE_H
#define FAST_MIDYNET_PYTHON_LABELED_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/random_graph/prior/python/prior.hpp"
#include "FastMIDyNet/random_graph/prior/labeled_degree.h"


namespace FastMIDyNet{

template <typename BaseClass = VertexLabeledDegreePrior>
class PyVertexLabeledDegreePrior: public PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, BaseClass> {
public:
    using PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, BaseClass>::PyVertexLabeledPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move);
    }
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move);
    }

    /* Overloaded abstract methods */
    void setState(const DegreeSequence& state) override {
        PYBIND11_OVERRIDE(void, BaseClass, setState, state);
    }
    const VertexLabeledDegreeCountsMap& getDegreeCounts() const override {
        PYBIND11_OVERRIDE(const VertexLabeledDegreeCountsMap&, BaseClass, getDegreeCounts, );
    }
    void checkSelfSafety() const override {
        PYBIND11_OVERRIDE_PURE(void, BaseClass, checkSelfSafety, );
    }
    void computationFinished() const override {
        PYBIND11_OVERRIDE_PURE(void, BaseClass, computationFinished, );
    }
};

}

#endif
