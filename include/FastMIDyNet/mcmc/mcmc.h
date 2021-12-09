#ifndef FAST_MIDYNET_MCMC_H
#define FAST_MIDYNET_MCMC_H

#include <sstream>
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

class MCMC{
protected:
    CallBackList m_callBacks;
    size_t m_numSteps;
    size_t m_numSweeps;
    double m_lastLogJointRatio;
    double m_lastLogAcceptance;
    bool m_isLastAccepted;
    bool m_hasState;
public:
    MCMC(std::vector<CallBack*> callbacks={}):
        m_callBacks(callbacks),
        m_numSteps(0),
        m_numSweeps(0),
        m_lastLogJointRatio(0),
        m_isLastAccepted(false),
        m_hasState(false) {}
    MCMC(const CallBackList& callbacks):
        m_callBacks(callbacks),
        m_numSteps(0),
        m_numSweeps(0),
        m_lastLogJointRatio(0),
        m_isLastAccepted(false),
        m_hasState(false) {}

    const double getLastLogJointRatio() const { return m_lastLogJointRatio; }
    const double getLastLogAcceptance() const { return m_lastLogAcceptance; }
    const bool isLastAccepted() const { return m_isLastAccepted; }
    const bool hasState() const { return m_hasState; }
    const size_t getNumSteps() const { return m_numSteps; }
    const size_t getNumSweeps() const { return m_numSweeps; }

    virtual double getLogLikelihood() const = 0 ;
    virtual double getLogPrior() = 0 ;
    virtual double getLogJoint() = 0 ;
    virtual void sample() = 0 ;

    void addCallBack(CallBack& callBack) { callBack.setUp(this); m_callBacks.pushBack(callBack); }
    void removeCallBack(size_t& idx) { m_callBacks.remove(idx); }

    virtual void setUp() { m_callBacks.setUp(this); m_numSteps = m_numSweeps = 0; }
    virtual void tearDown() { m_callBacks.tearDown(); m_numSteps = m_numSweeps = 0; }
    virtual void doMetropolisHastingsStep() = 0;
    void doMHSweep(size_t burn=1);


};

}

#endif