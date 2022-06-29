#ifndef FAST_MIDYNET_ACTION_HPP
#define FAST_MIDYNET_ACTION_HPP

#include "callback.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"

namespace FastMIDyNet{

class CheckConsistencyOnStep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onStepEnd() override { m_mcmcPtr->checkConsistency(); }
};

class CheckSafetyOnStep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onStepEnd() override { m_mcmcPtr->checkSafety(); }
};

class CheckConsistencyOnSweep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onSweepEnd() override { m_mcmcPtr->checkConsistency(); }
};

class CheckSafetyOnSweep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onSweepEnd() override { m_mcmcPtr->checkSafety(); }
};

}

#endif
