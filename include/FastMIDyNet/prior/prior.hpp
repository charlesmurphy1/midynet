#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include <functional>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template <typename STATE>
class Prior{
    public:
        const STATE& getState() const { return m_state; }
        void setState(const STATE& state) { m_state = state; }

        virtual void sampleState() = 0;
        virtual void samplePriors() = 0;
        virtual void sample() {
            processRecursiveFunction([&]() {
                    samplePriors();
                    sampleState();
                });
        }
        double getLogLikelihood() const { return getLogLikelihood(m_state); }
        virtual double getLogLikelihood(const STATE& state) const = 0;
        virtual double getLogPrior() = 0;

        double getLogJoint() {
            return processRecursiveFunction<double>(
                        [&]() { return getLogPrior()+getLogLikelihood(); },
                        0
                    );
        }

        virtual void checkSelfConsistency() const = 0;
        virtual void computationFinished() { m_isProcessed=false; }


        template<typename RETURN_TYPE>
        RETURN_TYPE processRecursiveFunction(const std::function<RETURN_TYPE()>& func, RETURN_TYPE init) {
            RETURN_TYPE ret = init;
            if (!m_isProcessed)
                ret = func();
            m_isProcessed=true;
            return ret;
        }
        void processRecursiveFunction(const std::function<void()>& func) {
            if (!m_isProcessed)
                func();
            m_isProcessed=true;
        }

    protected:
        STATE m_state;
        bool m_isProcessed = false;
};

}

#endif
