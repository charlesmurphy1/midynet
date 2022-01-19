#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include <functional>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template <typename StateType>
class Prior{
    public:
        Prior<StateType>(){}
        Prior<StateType>(const Prior<StateType>& other):
            m_state(other.m_state){}
        virtual ~Prior<StateType>(){}
        const Prior<StateType>& operator=(const Prior<StateType>& other){
            this->m_state = other.m_state;
            return *this;
        }

        const StateType& getState() const { return m_state; }
        StateType& getStateRef() const { return m_state; }
        virtual void setState(const StateType& state) { m_state = state; }
        const bool isRoot() const { return m_isRoot; }
        virtual void isRoot(bool condition) { m_isRoot = condition; }


        virtual void sampleState() = 0;
        virtual void samplePriors() = 0;
        const StateType& sample() {
            auto _func = [&]() {
                samplePriors();
                sampleState();
            };
            processRecursiveFunction(_func);
            return getState();
        }
        virtual const double getLogLikelihood() const = 0;
        virtual const double getLogPrior() const = 0;

        const double getLogJoint() const {
            auto _func = [&]() { return getLogPrior() + getLogLikelihood(); };
            double logJoint = processRecursiveConstFunction<double>(_func , 0);
            return logJoint;
        }

        virtual void checkSelfConsistency() const = 0;
        virtual void checkSafety() const = 0;
        virtual void computationFinished() const { m_isProcessed = false; }


        template<typename RETURN_TYPE>
        RETURN_TYPE processRecursiveConstFunction(const std::function<RETURN_TYPE()>& func, RETURN_TYPE init) const {
            RETURN_TYPE ret = init;
            if (!m_isProcessed)
                ret = func();

            if ( m_isRoot ) computationFinished();
            else m_isProcessed=true;
            return ret;
        }
        void processRecursiveConstFunction(const std::function<void()>& func) const {
            if (!m_isProcessed)
                func();
            m_isProcessed=true;
            if ( m_isRoot ) computationFinished();
            else m_isProcessed=true;
        }

        template<typename RETURN_TYPE>
        RETURN_TYPE processRecursiveFunction(const std::function<RETURN_TYPE()>& func, RETURN_TYPE init) const {
            RETURN_TYPE ret = init;
            if (!m_isProcessed)
                ret = func();

            if ( m_isRoot ) computationFinished();
            else m_isProcessed=true;
            return ret;
        }
        void processRecursiveFunction(const std::function<void()>& func) {
            if (!m_isProcessed)
                func();
            m_isProcessed=true;
            if ( m_isRoot ) computationFinished();
            else m_isProcessed=true;
        }

    protected:
        StateType m_state;
        mutable bool m_isRoot = true;
        mutable bool m_isProcessed = false;
};

}

#endif
