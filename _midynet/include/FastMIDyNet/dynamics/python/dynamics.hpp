#ifndef FAST_MIDYNET_PYTHON_DYNAMICS_HPP
#define FAST_MIDYNET_PYTHON_DYNAMICS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/python/rv.hpp"
// #include "FastMIDyNet/types.h"
// #include "FastMIDyNet/random_graph/random_graph.h"
// #include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = Dynamics>
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

template<typename BaseClass = BinaryDynamics>
class PyBinaryDynamics: public PyDynamics<BaseClass>{
public:
    using PyDynamics<BaseClass>::PyDynamics;
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
