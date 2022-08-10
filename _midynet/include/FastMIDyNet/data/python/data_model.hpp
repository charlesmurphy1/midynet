#ifndef FAST_MIDYNET_PYTHON_DATAMODEL_HPP
#define FAST_MIDYNET_PYTHON_DATAMODEL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/data/data_model.hpp"
#include "FastMIDyNet/data/dynamics/dynamics.hpp"
#include "FastMIDyNet/data/dynamics/binary_dynamics.hpp"


namespace FastMIDyNet{

template<typename GraphPrior, typename BaseClass = DataModel<GraphPrior>>
class PyDataModel: public PyNestedRandomVariable<BaseClass>{
protected:
    void applyGraphMoveToSelf(const GraphMove& move) override {
        PYBIND11_OVERRIDE_PURE(void, BaseClass, applyGraphMoveToSelf, move);
    }
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
    /* Pure abstract methods */
    void sampleState() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleState, ); }
    const double getLogLikelihood() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, );
    }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&move) const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move);
    }
    /* Abstract methods */
    void computeConsistentState() override { PYBIND11_OVERRIDE(void, BaseClass, computeConsistentState, ); }

    bool isSafe() const override { PYBIND11_OVERRIDE(bool, BaseClass, isSafe, ); }

};

template<typename GraphPrior, typename BaseClass = Dynamics<GraphPrior>>
class PyDynamics: public PyDataModel<GraphPrior, BaseClass>{
public:
    using PyDataModel<GraphPrior, BaseClass>::PyDataModel;

    /* Pure abstract methods */
    const double getTransitionProb(
        const VertexState& prevVertexState, const VertexState& nextVertexState, const VertexNeighborhoodState& neighborhoodState
    ) const {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getTransitionProb, prevVertexState, nextVertexState, neighborhoodState);
    }
    /* Abstract methods */
    const State getRandomState() const{ PYBIND11_OVERRIDE(const State, BaseClass, getRandomState, ); }

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
