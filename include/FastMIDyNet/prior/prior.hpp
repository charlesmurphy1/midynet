#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template <typename T>
class Prior{
    public:
        const T& getState() const { return m_state; }
        void setState(const T& state) { m_state = state; }

        virtual T sample() = 0;
        double getLogLikelihood() const { return getLogLikelihood(m_state); }
        virtual double getLogLikelihood(const T& state) const = 0;
        virtual double getLogPrior() const = 0;
        double getLogJoint() {
            double logLikelihood = 0;
            if (!isProcessed)
                logLikelihood = getLogPrior() + getLogLikelihood();
            isProcessed=true;
            return logLikelihood;
        }

        virtual void checkSelfConsistency() const = 0;
        virtual void computationFinished() { isProcessed=false; }

    protected:
        T m_state;
        bool isProcessed = false;
};

}

#endif
