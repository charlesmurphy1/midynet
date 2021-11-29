#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
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

        const size_t& getBlockCount() const { return m_blockPrior.getBlockCount(); }
        const size_t& getEdgeCount() const { return m_edgeCountPrior.getState(); }
        const std::vector<size_t>& getEdgeCountsInBlocks() { return m_edgeCountsInBlocks; }
        const BlockSequence& getBlockSequence() { return m_blockPrior.getState(); }

        void samplePriors() override { m_edgeCountPrior.sample(); m_blockPrior.sample(); }

        double getLogPrior() override { return m_edgeCountPrior.getLogJoint() + m_blockPrior.getLogJoint(); }

        virtual double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
        virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const = 0;

        double getLogPriorRatioFromGraphMove(const GraphMove& move) { return m_edgeCountPrior.getLogJointRatioFromGraphMove(move) + m_blockPrior.getLogJointRatioFromGraphMove(move); }
        double getLogPriorRatioFromBlockMove(const BlockMove& move) { return m_edgeCountPrior.getLogJointRatioFromBlockMove(move) + m_blockPrior.getLogJointRatioFromBlockMove(move); }

        double getLogJointRatioFromGraphMove(const GraphMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move); }, 0);
        }

        double getLogJointRatioFromBlockMove(const BlockMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0);
        }

        void applyGraphMoveToState(const GraphMove&);
        void applyBlockMoveToState(const BlockMove&);
        void applyGraphMove(const GraphMove& move){
            processRecursiveFunction( [&]() { m_edgeCountPrior.applyGraphMove(move); m_blockPrior.applyGraphMove(move); applyGraphMoveToState(move); } );
            #if DEBUG
            checkSelfConsistency();
            #endif
        }
        void applyBlockMove(const BlockMove& move) {
            processRecursiveFunction( [&]() { m_edgeCountPrior.applyBlockMove(move); m_blockPrior.applyBlockMove(move); applyBlockMoveToState(move); } );
            #if DEBUG
            checkSelfConsistency();
            #endif
        }

        void computationFinished() override {
            m_isProcessed = false;
            m_blockPrior.computationFinished();
            m_edgeCountPrior.computationFinished();
        }
        void checkSelfConsistencywithGraph() const;
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

class EdgeMatrixUniformPrior: public EdgeMatrixPrior {
public:
    using EdgeMatrixPrior::EdgeMatrixPrior;
    void sampleState();
    double getLogLikelihood() const {
        return getLogLikelihood(m_blockPrior.getBlockCount(), m_edgeCountPrior.getState());
    }
    double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;

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
