#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class NestedBlockCountPrior: public Prior<std::vector<size_t>> {
    public:
        void samplePriors() { }
        virtual double getLogLikelihoodFromState(const std::vector<size_t>&) const = 0;
        double getLogLikelihood() const { return getLogLikelihoodFromState(m_state); }
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatioFromNestedBlockMove(const NestedBlockMove& move) const {
            return getLogLikelihoodFromState(getStateAfterNestedBlockMove(move)) - getLogLikelihood();
        }
        double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; }
        double getLogJointRatioFromNestedBlockMove(const NestedBlockMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromNestedBlockMove(move); }, 0);
        }
        void applyGraphMove(const GraphMove& move) { }
        void applyNestedBlockMove(const NestedBlockMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterNestedBlockMove(move)); } );
        }
        std::vector<size_t> getStateAfterNestedBlockMove(const NestedBlockMove&) const;

};

}

#endif
