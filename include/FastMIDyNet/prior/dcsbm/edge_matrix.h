#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{

class EdgeMatrixPrior: public Prior< Matrix<size_t> >{
    EdgeCountPrior& m_edgeCountPrior;
    BlockPrior& m_blockPrior;

    public:
        EdgeMatrixPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior):
            m_edgeCountPrior(edgeCountPrior), m_blockPrior(blockPrior) { }

        const size_t& getEdgeCount() { return m_edgeCountPrior.getState(); }
        const BlockSequence& getBlockSequence() { return m_blockPrior.getState(); }

        double getLogPrior() { return m_edgeCountPrior.getLogJoint() + m_blockPrior.getLogJoint(); }
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
};


class EdgeMatrixUniformPrior: public EdgeMatrixPrior {
    public:
        Matrix<size_t> sample();
        void sampleState();
        double getLogLikelihoodRatio(const GraphMove&) const;
        double getLogLikelihoodRatio(const BlockMove&) const;
        void applyMove(const GraphMove&);
        void applyMove(const BlockMove&);
};

} // namespace FastMIDyNet

#endif
