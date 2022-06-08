#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include <functional>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template <typename StateType>
class Prior: public NestedRandomVariable{
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

        virtual void sampleState() = 0;
        virtual void samplePriors() = 0;
        const StateType& sample() {
            processRecursiveFunction([&]() {
                samplePriors();
                sampleState();
            });
            return getState();
        }
        virtual const double getLogLikelihood() const = 0;
        virtual const double getLogPrior() const = 0;

        void applyGraphMove(const GraphMove& move) {
            processRecursiveFunction([&](){_applyGraphMove(move);});
            #if DEBUG
            checkConsistency();
            #endif
        }
        void applyBlockMove(const BlockMove& move) {
            processRecursiveFunction([&](){_applyBlockMove(move);});
            #if DEBUG
            checkConsistency();
            #endif
        }
        const double getLogJointRatioFromGraphMove(const GraphMove& move) const {
            return processRecursiveConstFunction<double>([&](){ return _getLogJointRatioFromGraphMove(move);}, 0);
        }
        const double getLogJointRatioFromBlockMove(const BlockMove& move) const {
            return processRecursiveConstFunction<double>([&](){ return _getLogJointRatioFromBlockMove(move);}, 0);
        }




        const double getLogJoint() const {
            auto _func = [&]() { return getLogPrior() + getLogLikelihood(); };
            double logJoint = processRecursiveConstFunction<double>(_func , 0);
            return logJoint;
        }
    protected:

        virtual void onBlockCreation(const BlockMove&) { };
        virtual void onBlockDeletion(const BlockMove&) { };
        virtual void _applyGraphMove(const GraphMove&) = 0;
        virtual void _applyBlockMove(const BlockMove&) = 0;
        virtual const double _getLogJointRatioFromGraphMove(const GraphMove&) const = 0;
        virtual const double _getLogJointRatioFromBlockMove(const BlockMove&) const = 0;
        StateType m_state;
};

}

#endif
