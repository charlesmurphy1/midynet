#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include "FastMIDyNet/types.h"


namespace FastMIDyNet{

template <typename T>
class Prior{
    public:
        Prior(RNG& rng) : m_rng(rng) {};
        const T& getState() { return m_state; }
        void setState(const T& state) { m_state = state; }

        virtual T sample() = 0;
        virtual double getLogLikelihood(const T& prior) const = 0;
        virtual double getLogLikelihood() const { return getLogLikelihood(m_state); }
        double getLogPrior() const { return 0; }
        double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
        double getLogLikelihoodRatio(const GraphMove& move) const {
            T newState = getStateAfterMove(move);
            return getLogLikelihood(newState) - getLogLikelihood();
        }

        void applyMove(const GraphMove&) { setState(getStateAfterMove(move)); }
        virtual T getStateAfterMove(const GraphMove&) = 0;

        virtual void checkConsistency() = 0;


    protected:
        T m_state;
        RNG m_rng;
};

}

#endif
