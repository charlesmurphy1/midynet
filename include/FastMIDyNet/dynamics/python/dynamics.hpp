#ifndef FAST_MIDYNET_PYTHON_DYNAMICS_HPP
#define FAST_MIDYNET_PYTHON_DYNAMICS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #include "FastMIDyNet/types.h"
// #include "FastMIDyNet/random_graph/random_graph.h"
// #include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = Dynamics>
class PyDynamics: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */
    double getTransitionProb(
        VertexState prevVertexState,
        VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const override {
        PYBIND11_OVERRIDE_PURE(double, BaseClass, getTransitionProb, prevVertexState, nextVertexState, neighborhoodState);
    }

    /* Abstract methods */
    const State getRandomState() override{
        PYBIND11_OVERRIDE(const State, BaseClass, getRandomState, );
    };


};

template<typename BaseClass = BinaryDynamics>
class PyBinaryDynamics: public PyDynamics<BaseClass>{
public:
    using PyDynamics<BaseClass>::PyDynamics;
    /* Pure abstract methods */
    double getActivationProb(const VertexNeighborhoodState& neighborState) const override {
        PYBIND11_OVERRIDE_PURE(double, BaseClass, getActivationProb, neighborState);
    }
    double getDeactivationProb(const VertexNeighborhoodState& neighborState) const override {
        PYBIND11_OVERRIDE_PURE(double, BaseClass, getDeactivationProb, neighborState);
    }
    /* Abstract methods */
    double getTransitionProb(
        VertexState prevVertexState,
        VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState
    ) const override {
        PYBIND11_OVERRIDE(double, BaseClass, getTransitionProb, prevVertexState, nextVertexState, neighborhoodState);
    }
};

}

#endif
