#ifndef FAST_MIDYNET_PYTHON_SAMPLER_HPP
#define FAST_MIDYNET_PYTHON_SAMPLER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"


namespace py = pybind11;
namespace FastMIDyNet{


template<typename BaseClass = VertexSampler>
class PyVertexSampler: public BaseClass{
public:
    using BaseClass::BaseClass;

    /* Pure abstract methods */
    BaseGraph::VertexIndex sample() const override { PYBIND11_OVERRIDE_PURE(const BaseGraph::VertexIndex, BaseClass, sample, ) ;}
    bool contains(const BaseGraph::VertexIndex& vertex) const override { PYBIND11_OVERRIDE_PURE(bool, BaseClass, contains, vertex) ;}
    void onVertexInsertion(const BaseGraph::VertexIndex& vertex) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, onVertexInsertion, vertex) ;}
    void onVertexErasure(const BaseGraph::VertexIndex& vertex) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, onVertexErasure, vertex) ;}
    void onEdgeInsertion(const BaseGraph::Edge& edge, double weight) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, onEdgeInsertion, edge, weight) ;}
    void onEdgeErasure(const BaseGraph::Edge& edge) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, onEdgeErasure, edge) ;}
    void onEdgeAddition(const BaseGraph::Edge& edge) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, onEdgeAddition, edge) ;}
    void onEdgeRemoval(const BaseGraph::Edge& edge) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, onEdgeRemoval, edge) ;}
    const double getVertexWeight(const BaseGraph::VertexIndex& vertexIdx) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getVertexWeight, vertexIdx) ;}
    const double getTotalWeight() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getTotalWeight, ) ;}
    const size_t getSize() const override { PYBIND11_OVERRIDE_PURE(const size_t, BaseClass, getSize, ) ;}

    /* Abstract & overloaded methods */
    void checkSafety() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSafety, ) ;}
    void clear() override { PYBIND11_OVERRIDE(void, BaseClass, clear, ) ;}
};

}

#endif
