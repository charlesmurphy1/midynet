#ifndef FAST_MIDYNET_PRIOR_HPP
#define FAST_MIDYNET_PRIOR_HPP


#include <functional>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template <typename StateType>
class Prior: public NestedRandomVariable{
protected:
    virtual void _applyGraphMove(const GraphMove&) = 0;
    virtual const double _getLogJointRatioFromGraphMove(const GraphMove&) const = 0;
    StateType m_state;
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
        NestedRandomVariable::processRecursiveFunction([&]() {
            samplePriors();
            sampleState();
        });
        return getState();
    }
    virtual const double getLogLikelihood() const = 0;
    virtual const double getLogPrior() const = 0;

    void applyGraphMove(const GraphMove& move) {
        NestedRandomVariable::processRecursiveFunction([&](){_applyGraphMove(move);});
        #if DEBUG
        checkConsistency();
        #endif
    }
    const double getLogJointRatioFromGraphMove(const GraphMove& move) const {
        return NestedRandomVariable::processRecursiveConstFunction<double>([&](){ return _getLogJointRatioFromGraphMove(move);}, 0);
    }

    const double getLogJoint() const {
        auto _func = [&]() { return getLogPrior() + getLogLikelihood(); };
        double logJoint = processRecursiveConstFunction<double>(_func , 0);
        return logJoint;
    }
};

template <typename StateType, typename Label>
class VertexLabeledPrior: public Prior<StateType>{
protected:
    virtual void _applyLabelMove(const LabelMove<Label>&) = 0;
    virtual const double _getLogJointRatioFromLabelMove(const LabelMove<Label>&) const = 0;
public:
    using Prior<StateType>::Prior;

    void applyLabelMove(const LabelMove<Label>& move) {
        NestedRandomVariable::processRecursiveFunction([&](){_applyLabelMove(move);});
        #if DEBUG
        checkConsistency();
        #endif
    }
    const double getLogJointRatioFromLabelMove(const LabelMove<Label>& move) const {
        return NestedRandomVariable::processRecursiveConstFunction<double>([&](){ return _getLogJointRatioFromLabelMove(move);}, 0.);
    }
};

template <typename StateType>
class SBMPrior: public VertexLabeledPrior<StateType, BlockIndex>{
public:
    using VertexLabeledPrior<StateType, BlockIndex>::VertexLabeledPrior;
};

}

#endif
