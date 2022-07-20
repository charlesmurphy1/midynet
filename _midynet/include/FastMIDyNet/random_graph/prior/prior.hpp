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
    virtual void _samplePriors() = 0;
    virtual void _applyGraphMove(const GraphMove&) = 0;
    virtual const double _getLogPrior() const = 0;
    virtual const double _getLogPriorRatioFromGraphMove(const GraphMove&) const = 0;
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
    void samplePriors() {
        NestedRandomVariable::processRecursiveFunction([&]() { _samplePriors(); });
    };
    const StateType& sample() {
        samplePriors();
        sampleState();
        return getState();
    }
    virtual const double getLogLikelihood() const = 0;
    const double getLogPrior() const {
        return processRecursiveConstFunction<double>([&]() { return _getLogPrior(); }, 0);
    }

    void applyGraphMove(const GraphMove& move) {
        NestedRandomVariable::processRecursiveFunction([&](){_applyGraphMove(move);});
        #if DEBUG
        checkConsistency();
        #endif
    }

    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const = 0;
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const {
        return processRecursiveConstFunction<double>([&]() {
            return _getLogPriorRatioFromGraphMove(move);
        }, 0);
    }

    const double getLogJointRatioFromGraphMove(const GraphMove& move) const {
        return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move);
    }

    const double getLogJoint() const {
        return getLogPrior() + getLogLikelihood();
    }
};

template <typename StateType, typename Label>
class VertexLabeledPrior: public Prior<StateType>{
protected:
    virtual void _applyLabelMove(const LabelMove<Label>&) = 0;
    virtual const double _getLogPriorRatioFromLabelMove(const LabelMove<Label>&) const = 0;
public:
    using Prior<StateType>::Prior;

    void applyLabelMove(const LabelMove<Label>& move) {
        NestedRandomVariable::processRecursiveFunction([&](){_applyLabelMove(move);});
        #if DEBUG
        checkConsistency();
        #endif
    }
    virtual const double getLogLikelihoodRatioFromLabelMove(const LabelMove<Label>& move) const = 0;
    const double getLogPriorRatioFromLabelMove(const LabelMove<Label>& move) const {
        return NestedRandomVariable::processRecursiveConstFunction<double>([&](){
            return _getLogPriorRatioFromLabelMove(move);
        }, 0.);
    }
    const double getLogJointRatioFromLabelMove(const LabelMove<Label>& move) const {
        return getLogLikelihoodRatioFromLabelMove(move) + getLogPriorRatioFromLabelMove(move);
    }
};

template<typename StateType>
using BlockLabeledPrior = VertexLabeledPrior<StateType, BlockIndex>;

}

#endif
