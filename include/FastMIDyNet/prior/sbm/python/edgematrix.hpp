#ifndef FAST_MIDYNET_PYTHON_EDGEMATRIX_H
#define FAST_MIDYNET_PYTHON_EDGEMATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"


namespace FastMIDyNet{

template <typename EdgeMatrixPriorBaseClass = EdgeMatrixPrior>
class PyEdgeMatrixPrior: public PyPrior<std::vector<std::vector<size_t>>, EdgeMatrixPriorBaseClass> {
public:
    using PyPrior<std::vector<std::vector<size_t>>, EdgeMatrixPriorBaseClass>::PyPrior;
    /* Pure abstract methods */
    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(double, EdgeMatrixPriorBaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(double, EdgeMatrixPriorBaseClass, getLogLikelihoodRatioFromBlockMove, move); }

    /* Overloaded abstract methods */
};

}

#endif



// class EdgeMatrixPrior: public Prior< EdgeMatrix >{
//     protected:
//         const MultiGraph* m_graph;
//         EdgeCountPrior& m_edgeCountPrior;
//         BlockPrior& m_blockPrior;
//         std::vector<size_t> m_edgeCountsInBlocks;
//
//         void createBlock();
//         void destroyBlock(const BlockIndex&);
//         void moveEdgeCountsInBlocks(const BlockMove& move);
//     public:
//         EdgeMatrixPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior):
//             m_edgeCountPrior(edgeCountPrior), m_blockPrior(blockPrior) {}
//
//
//         void setGraph(const MultiGraph& graph);
//         const MultiGraph& getGraph() { return *m_graph; }
//         void setState(const EdgeMatrix&) override;
//
//         const size_t& getBlockCount() const { return m_blockPrior.getBlockCount(); }
//         const size_t& getEdgeCount() const { return m_edgeCountPrior.getState(); }
//         const std::vector<size_t>& getEdgeCountsInBlocks() { return m_edgeCountsInBlocks; }
//         const BlockSequence& getBlocks() { return m_blockPrior.getState(); }
//
//         void samplePriors() override { m_edgeCountPrior.sample(); m_blockPrior.sample(); }
//
//         double getLogPrior() override { return m_edgeCountPrior.getLogJoint() + m_blockPrior.getLogJoint(); }
//
//         virtual double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
//         virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const = 0;
//
//         double getLogPriorRatioFromGraphMove(const GraphMove& move) { return m_edgeCountPrior.getLogJointRatioFromGraphMove(move) + m_blockPrior.getLogJointRatioFromGraphMove(move); }
//         double getLogPriorRatioFromBlockMove(const BlockMove& move) { return m_edgeCountPrior.getLogJointRatioFromBlockMove(move) + m_blockPrior.getLogJointRatioFromBlockMove(move); }
//
//         double getLogJointRatioFromGraphMove(const GraphMove& move) {
//             return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move); }, 0);
//         }
//
//         double getLogJointRatioFromBlockMove(const BlockMove& move) {
//             return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0);
//         }
//
//         void applyGraphMoveToState(const GraphMove&);
//         void applyBlockMoveToState(const BlockMove&);
//         void applyGraphMove(const GraphMove& move){
//             processRecursiveFunction( [&]() { m_edgeCountPrior.applyGraphMove(move); m_blockPrior.applyGraphMove(move); applyGraphMoveToState(move); } );
//             #if DEBUG
//             checkSelfConsistency();
//             #endif
//         }
//         void applyBlockMove(const BlockMove& move) {
//             processRecursiveFunction( [&]() { m_edgeCountPrior.applyBlockMove(move); m_blockPrior.applyBlockMove(move); applyBlockMoveToState(move); } );
//             #if DEBUG
//             checkSelfConsistency();
//             #endif
//         }
//
//         void computationFinished() override {
//             m_isProcessed = false;
//             m_blockPrior.computationFinished();
//             m_edgeCountPrior.computationFinished();
//         }
//         void checkSelfConsistencywithGraph() const;
//         void checkSelfConsistency() const override;
// };
