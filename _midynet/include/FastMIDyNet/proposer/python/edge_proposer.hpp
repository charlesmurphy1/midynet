#ifndef FAST_MIDYNET_PYTHON_EDGE_PROPOSER_HPP
#define FAST_MIDYNET_PYTHON_EDGE_PROPOSER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"
#include "FastMIDyNet/proposer/edge_proposer/labeled_edge_proposer.h"
#include "FastMIDyNet/proposer/edge_proposer/labeled_hinge_flip.h"

// namespace py = pybind11;
namespace FastMIDyNet{

template<typename BaseClass = EdgeProposer>
class PyEdgeProposer: public PyProposer<GraphMove, BaseClass>{
public:
    using PyProposer<GraphMove, BaseClass>::PyProposer;
    ~PyEdgeProposer() override = default;

    /* Pure abstract methods */
    GraphMove proposeRawMove() const override { PYBIND11_OVERRIDE_PURE(GraphMove, BaseClass, proposeRawMove, ); }
    const double getLogProposalProbRatio(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbRatio, move); }

    /* Abstract & overloaded methods */
    void setUp(const RandomGraph& randomGraph) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setUp, randomGraph); }
    void setUpFromGraph(const MultiGraph& graph) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setUpFromGraph, graph); }
    void applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void applyBlockMove(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyBlockMove, move); }
    GraphMove proposeMove() const override { PYBIND11_OVERRIDE(GraphMove, BaseClass, proposeMove, ); }
};

template<typename BaseClass = HingeFlipProposer>
class PyHingeFlipProposer: public PyEdgeProposer<BaseClass>{
public:
    using PyEdgeProposer<BaseClass>::PyEdgeProposer;
    /* Pure abstract methods */
    const double getLogVertexWeightRatio(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getVertexWeightRatio, move); }

    /* Abstract & overloaded methods */
    ~PyHingeFlipProposer() override = default;
};

template<typename BaseClass = SingleEdgeProposer>
class PySingleEdgeProposer: public PyEdgeProposer<BaseClass>{
public:
    using PyEdgeProposer<BaseClass>::PyEdgeProposer;
    /* Pure abstract methods */
    const double getLogProposalProbRatio(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbRatio, move); }

    /* Abstract & overloaded methods */
    ~PySingleEdgeProposer() override = default;
};


template<typename BaseClass = LabeledEdgeProposer>
class PyLabeledEdgeProposer: public PyEdgeProposer<BaseClass>{
public:
    using PyEdgeProposer<BaseClass>::PyEdgeProposer;
    /* Pure abstract methods */

    /* Abstract & overloaded methods */
    void setUp( const RandomGraph& randomGraph ) override { PYBIND11_OVERRIDE(void, BaseClass, setUp, randomGraph); }
    void setUpFromGraph(const MultiGraph& graph) override { PYBIND11_OVERRIDE(void, BaseClass, setUpFromGraph, graph); }
    void onLabelCreation(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, onLabelCreation, move); }
    void onLabelDeletion(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, onLabelDeletion, move); }
};



template<typename BaseClass = LabeledHingeFlipProposer>
class PyLabeledHingeFlipProposer: public PyEdgeProposer<BaseClass>{
public:
    using PyEdgeProposer<BaseClass>::PyEdgeProposer;
    /* Pure abstract methods */
    const double getLogProposalProbRatio(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbRatio, move); }
    VertexSampler* constructVertexSampler() const override { PYBIND11_OVERRIDE_PURE(VertexSampler*, BaseClass, constructVertexSampler, ); }

    /* Abstract & overloaded methods */
};

}

#endif
