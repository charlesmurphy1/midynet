#ifndef FAST_MIDYNET_PYTHON_NESTEDLABELGRAPH_H
#define FAST_MIDYNET_PYTHON_NESTEDLABELGRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/prior/python/label_graph.hpp"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"


namespace FastMIDyNet{

template <typename BaseClass = NestedLabelGraphPrior>
class PyNestedLabelGraphPrior: public PyLabelGraphPrior< BaseClass > {
public:
    using PyLabelGraphPrior<BaseClass>::PyLabelGraphPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const LabelGraph sampleState(Level level) const override { PYBIND11_OVERRIDE_PURE(const LabelGraph, BaseClass, sampleState, level); }
    const double getLogLikelihoodAtLevel(Level level) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodAtLevel, level); }

    /* Overloaded abstract methods */
};

}

#endif
