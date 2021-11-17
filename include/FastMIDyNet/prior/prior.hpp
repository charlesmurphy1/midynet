#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template <typename T>
class Prior{
    public:
        const T& getState() { return m_state; }
        void setState(const T& state) { m_state = state; }

        virtual T sample() = 0;
        double getLogLikelihood() const { return getLogLikelihood(m_state); }
        virtual double getLogLikelihood(size_t state) const = 0;
        virtual double getLogPrior() const = 0;
        double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }

        virtual void checkSelfConsistency() = 0;

    protected:
        T m_state;
};

}

#endif
