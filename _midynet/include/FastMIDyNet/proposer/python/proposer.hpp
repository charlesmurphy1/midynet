#ifndef FAST_MIDYNET_PYTHON_PROPOSER_HPP
#define FAST_MIDYNET_PYTHON_PROPOSER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/proposer/edge_proposer/vertex_sampler.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/random_graph/sbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename MoveType,typename BaseClass = Proposer<MoveType>>
class PyProposer: public BaseClass{
public:
    using BaseClass::BaseClass;

    /* Pure abstract methods */
    MoveType proposeMove() override { PYBIND11_OVERRIDE_PURE(MoveType, BaseClass, proposeMove, ); }
    double getLogProposalProbRatio(const MoveType& move) const override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogProposalProbRatio, move); }
    void updateProbabilities(const MoveType& move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, updateProbabilities, move); }

    /* Abstract & overloaded methods */
};

template<typename BaseClass = EdgeProposer>
class PyEdgeProposer: public PyProposer<GraphMove, BaseClass>{
public:
    using PyProposer<GraphMove, BaseClass>::PyProposer;
    /* Pure abstract methods */
    void setUp(const RandomGraph& randomGraph) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setUp, randomGraph); }

    /* Abstract & overloaded methods */
    bool setAcceptIsolated(bool accept) override { PYBIND11_OVERRIDE(bool, BaseClass, setAcceptIsolated, accept); }
};

template<typename BaseClass = BlockProposer>
class PyBlockProposer: public PyProposer<BlockMove, BaseClass>{
public:
    using PyProposer<BlockMove, BaseClass>::PyProposer;

    /* Pure abstract methods */
    void setUp(const StochasticBlockModelFamily& randomGraph) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setUp, randomGraph); }

    /* Abstract & overloaded methods */
};

template<typename BaseClass = VertexSampler>
class PyVertexSampler: public BaseClass{
public:
    using BaseClass::BaseClass;

    /* Pure abstract methods */
    const BaseGraph::VertexIndex sample() override { PYBIND11_OVERRIDE_PURE(const BaseGraph::VertexIndex, BaseClass, sample, ) ;}
    void setUp(const MultiGraph& graph) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setUp, graph) ;}
    void update(const GraphMove& move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, update, move) ;}
    double getLogProposalProbRatio(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogProposalProbRatio, move) ;}

    /* Abstract & overloaded methods */
};

}

#endif
