#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/functions.h"


namespace FastMIDyNet{

class EdgeMatrixPrior: public Prior< Matrix<size_t> >{
    public:
        EdgeMatrixPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior):
            m_edgeCountPrior(edgeCountPrior), m_blockPrior(blockPrior) {}

        void setGraph(const MultiGraph& graph);
        void setState(const Matrix<size_t>&) override;

        void samplePriors() override { m_edgeCountPrior.sample(); m_blockPrior.sample(); }

        const size_t& getEdgeCount() const { return m_edgeCountPrior.getState(); }
        const std::vector<size_t>& getEdgesInBlock() { return m_edgeCountInBlocks; }
        const BlockSequence& getBlockSequence() { return m_blockPrior.getState(); }

        double getLogPrior() override { return m_edgeCountPrior.getLogJoint() + m_blockPrior.getLogJoint(); }
        virtual double getLogLikelihoodRatio(const GraphMove&) const = 0;
        virtual double getLogLikelihoodRatio(const BlockMove&) const = 0;

        double getLogJointRatio(const GraphMove& move) {
            return processRecursiveFunction<double>( [&]() {
                        return getLogLikelihoodRatio(move) +
                            m_edgeCountPrior.getLogJointRatio(move) + m_blockPrior.getLogJointRatio(move);
                    },
                    0
                );
        }

        double getLogJointRatio(const BlockMove& move) {
            return processRecursiveFunction<double>( [&]() {
                        return getLogLikelihoodRatio(move) +
                            m_edgeCountPrior.getLogJointRatio(move) + m_blockPrior.getLogJointRatio(move);
                    },
                    0
                );
        }

        virtual void applyMove(const GraphMove&) = 0;
        virtual void applyMove(const BlockMove&) = 0;

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
        std::vector<size_t> m_edgeCountInBlocks;

        void createBlock();
        void destroyBlock(const BlockIndex&);
        void moveEdgesInBlocks(const BlockMove& move);
};


class EdgeMatrixUniformPrior: public EdgeMatrixPrior {
    public:
        void sampleState();
        double getLogLikelihood() {
            return getLogLikelihood(m_blockPrior.getBlockCount(), m_edgeCountPrior.getState());
        }
        double getLogLikelihoodRatio(const GraphMove&) const;
        double getLogLikelihoodRatio(const BlockMove&) const;
        void applyMove(const GraphMove&);
        void applyMove(const BlockMove&);

    protected:
        double getLikelihoodRatio(size_t blockCountAfter, size_t edgeNumberAfter) const {
            return getLogLikelihood(m_edgeCountPrior.getState(), m_blockPrior.getBlockCount())
                - getLogLikelihood(blockCountAfter, edgeNumberAfter);
        }
        double getLogLikelihood(size_t blockNumber, size_t edgeNumber) const {
            return logMultisetCoefficient(
                    blockNumber*(blockNumber+1)/2,
                    edgeNumber);
        }
};

} // namespace FastMIDyNet

#endif
