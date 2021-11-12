#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{
    template <typename T>
    class Prior{
    public:

        Prior(const Prior& prior, RNG rng) : m_prior(prior), m_rng(rng) {};
        const T& getState() { return m_state; }
        void setState(const T& state) { m_state = state; }


        virtual const T& sample() = 0;
        virtual const double& getLogLikelihood() const = 0;
        virtual const double& getLogPrior() const = 0;
        const double& getLogJoint() const { return getLogPrior() + getLogLikelihood(); }

        void applyMove(const GraphMove&);
        void applyMove(const PriorMove&);

    protected:
        Prior& m_prior;
        T m_state;
        RNG m_rng;
    };
}



#endif
