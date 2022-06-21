#ifndef FAST_MIDYNET_PYTHON_DYNAMICS_HPP
#define FAST_MIDYNET_PYTHON_DYNAMICS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/dynamics/dynamics.hpp"
#include "FastMIDyNet/dynamics/binary_dynamics.hpp"
// #include "FastMIDyNet/types.h"
// #include "FastMIDyNet/random_graph/random_graph.h"
// #include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename GraphPrior, typename BaseClass = Dynamics<GraphPrior>>
class PyDynamics: public PyNestedRandomVariable<BaseClass>{
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
    /* Pure abstract methods */
    const double getTransitionProb(
        VertexState prevVertexState,
        VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getTransitionProb, prevVertexState, nextVertexState, neighborhoodState);
    }

    /* Abstract methods */
    const State getRandomState() const override{ PYBIND11_OVERRIDE(const State, BaseClass, getRandomState, ); }

};

template<typename GraphPrior, typename BaseClass = BinaryDynamics<GraphPrior>>
class PyBinaryDynamics: public PyDynamics<GraphPrior, BaseClass>{
public:
    using PyDynamics<GraphPrior, BaseClass>::PyDynamics;
    /* Pure abstract methods */
    const double getActivationProb(const VertexNeighborhoodState& neighborState) const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getActivationProb, neighborState);
    }
    const double getDeactivationProb(const VertexNeighborhoodState& neighborState) const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getDeactivationProb, neighborState);
    }
    /* Abstract methods */
};

}

#endif
