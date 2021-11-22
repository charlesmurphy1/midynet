#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"


namespace FastMIDyNet{

class EdgeMatrixPrior: public Prior< EdgeMatrix >{
    public:
        EdgeMatrixPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior):
            m_edgeCountPrior(edgeCountPrior), m_blockPrior(blockPrior) {}

        void setGraph(const MultiGraph& graph);
        void setState(const EdgeMatrix&) override;

        const size_t& getEdgeCount() { return m_edgeCountPrior.getState(); }
        const size_t& getBlockCount() { return m_blockPrior.getBlockCount(); }
        const size_t& getEdgeCount() const { return m_edgeCountPrior.getState(); }
        const std::vector<size_t>& getEdgeCountsInBlock() { return m_edgeCountsInBlocks; }
        const BlockSequence& getBlockSequence() { return m_blockPrior.getState(); }

        void samplePriors() override { m_edgeCountPrior.sample(); m_blockPrior.sample(); }

        double getLogPrior() override { return m_edgeCountPrior.getLogJoint() + m_blockPrior.getLogJoint(); }

        virtual double getLogLikelihoodRatio(const GraphMove&) const = 0;
        virtual double getLogLikelihoodRatio(const BlockMove&) const = 0;

        double getLogPriorRatio(const GraphMove& move) { return m_edgeCountPrior.getLogJointRatio(move) + m_blockPrior.getLogJointRatio(move); }
        double getLogPriorRatio(const BlockMove& move) { return m_edgeCountPrior.getLogJointRatio(move) + m_blockPrior.getLogJointRatio(move); }

        double getLogJointRatio(const GraphMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }, 0);
        }

        double getLogJointRatio(const BlockMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }, 0);
        }

        void applyMoveToState(const GraphMove&);
        void applyMoveToState(const BlockMove&);
        void applyMove(const GraphMove& move){
            processRecursiveFunction( [&]() { m_edgeCountPrior.applyMove(move); m_blockPrior.applyMove(move); applyMoveToState(move); } );
        }
        void applyMove(const BlockMove& move) {
            processRecursiveFunction( [&]() { m_edgeCountPrior.applyMove(move); m_blockPrior.applyMove(move); applyMoveToState(move); } );
        }

        void computationFinished() override {
            m_isProcessed = false;
            m_blockPrior.computationFinished();
            m_edgeCountPrior.computationFinished();
        }
        void checkSelfConsistency() const override;

    protected:
        const MultiGraph* m_graph;
        EdgeCountPrior& m_edgeCountPrior;
        BlockPrior& m_blockPrior;
        std::vector<size_t> m_edgeCountsInBlocks;

        void createBlock();
        void destroyBlock(const BlockIndex&);
        void moveEdgeCountsInBlocks(const BlockMove& move);
};


class EdgeMatrixDeltaPrior: public EdgeMatrixPrior{
    EdgeMatrix m_edgeMatrix;
    EdgeCountDeltaPrior m_edgeCountDeltaPrior;
public:
    EdgeMatrixDeltaPrior(EdgeMatrix edgeMatrix, BlockPrior& blockPrior):
    m_edgeMatrix(m_edgeMatrix),
    m_edgeCountDeltaPrior( sumElementsOfMatrix(m_edgeMatrix, (size_t) 0) ),
    EdgeMatrixPrior(m_edgeCountDeltaPrior, blockPrior) { setState(m_edgeMatrix); }

    void sampleState() const { }
    double getLogLikelihoodRatio(const GraphMove&) const;
    double getLogLikelihoodRatio(const BlockMove&) const;
};

class EdgeMatrixUniformPrior: public EdgeMatrixPrior {
public:
    using EdgeMatrixPrior::EdgeMatrixPrior;
    void sampleState();
    double getLogLikelihood() const {
        return getLogLikelihood(m_blockPrior.getBlockCount(), m_edgeCountPrior.getState());
    }
    double getLogLikelihoodRatio(const GraphMove&) const;
    double getLogLikelihoodRatio(const BlockMove&) const;

protected:
    double getLikelihoodRatio(size_t blockCountAfter, size_t edgeNumberAfter) const {
        return getLogLikelihood(m_edgeCountPrior.getState(), m_blockPrior.getBlockCount())
            - getLogLikelihood(blockCountAfter, edgeNumberAfter);
    }
    double getLogLikelihood(size_t blockNumber, size_t edgeNumber) const {
        return -logMultisetCoefficient(
                blockNumber*(blockNumber+1)/2,
                edgeNumber);
    }
};

} // namespace FastMIDyNet

#endif
