#ifndef FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP
#define FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = RandomGraph>
class PyRandomGraph: public PyNestedRandomVariable<BaseClass>{
protected:
    void _applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;

    /* Pure abstract methods */
    void sample() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sample, ); }
    const double getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPrior, ); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPriorRatioFromGraphMove, move); }

    const size_t& getEdgeCount() const override  {
        PYBIND11_OVERRIDE_PURE(const size_t&, BaseClass, getEdgeCount, );
    }

    /* Abstract methods */
    void setGraph(const MultiGraph& graph) override { PYBIND11_OVERRIDE(void, BaseClass, setGraph, graph); }
    const bool isCompatible(const MultiGraph& graph) const override { PYBIND11_OVERRIDE(bool, BaseClass, isCompatible, graph); }
    bool isSafe() const override { PYBIND11_OVERRIDE(bool, BaseClass, isSafe, ); }
};

template<typename Label, typename BaseClass = VertexLabeledRandomGraph<Label>>
class PyVertexLabeledRandomGraph: public PyRandomGraph<BaseClass>{
protected:
    void _applyLabelMove(const LabelMove<Label>& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyLabelMove, move); }
public:
    using PyRandomGraph<BaseClass>::PyRandomGraph;

    /* Pure abstract methods */
    void setLabels(const std::vector<BlockIndex>& labels) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setLabels, labels); }
    void sampleLabels () override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleLabels,); }
    const double getLogLikelihoodRatioFromLabelMove (const LabelMove<Label>& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }
    const double getLogPriorRatioFromLabelMove (const LabelMove<Label>& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPriorRatioFromLabelMove, move); }

    const std::vector<Label>& getVertexLabels() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<Label>&, BaseClass, getVertexLabels, );
    }
    const CounterMap<Label>& getLabelCounts() const override  {
        PYBIND11_OVERRIDE_PURE(const CounterMap<Label>&, BaseClass, getLabelCounts, );
    }
    const CounterMap<Label>& getEdgeLabelCounts() const override  {
        PYBIND11_OVERRIDE_PURE(const CounterMap<Label>&, BaseClass, getEdgeLabelCounts, );
    }
    const MultiGraph& getLabelGraph() const override  {
        PYBIND11_OVERRIDE_PURE(const MultiGraph&, BaseClass, getLabelGraph, );
    }

    /* Abstract methods */
};

}

#endif
