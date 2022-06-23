#ifndef FAST_MIDYNET_MCMC_H
#define FAST_MIDYNET_MCMC_H

#include <sstream>
#include <tuple>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"


namespace FastMIDyNet{

class MCMC: public NestedRandomVariable{
protected:
    CallBackMap<MCMC> m_mcmcCallBacks;
    size_t m_numSteps;
    size_t m_numSweeps;
    mutable double m_lastLogJointRatio;
    mutable double m_lastLogAcceptance;
    mutable bool m_isLastAccepted;
    double m_betaLikelihood, m_betaPrior;
    mutable std::uniform_real_distribution<double> m_uniform;
public:
    MCMC(double betaPrior=1, double betaLikelihood=1):
        m_betaPrior(betaPrior),
        m_betaLikelihood(betaLikelihood),
        m_numSteps(0),
        m_numSweeps(0),
        m_lastLogJointRatio(0),
        m_isLastAccepted(false),
        m_uniform(0, 1) {}

    const double getLastLogJointRatio() const { return m_lastLogJointRatio; }
    const double getLastLogAcceptance() const { return m_lastLogAcceptance; }
    const bool isLastAccepted() const { return m_isLastAccepted; }
    const size_t getNumSteps() const { return m_numSteps; }
    const size_t getNumSweeps() const { return m_numSweeps; }

    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }
    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    virtual void sample() = 0;
    virtual void samplePrior() = 0;
    virtual const double getLogLikelihood() const = 0 ;
    virtual const double getLogPrior() const = 0 ;
    virtual const double getLogJoint() const = 0 ;

    void insertCallBack(std::pair<std::string, CallBack<MCMC>*> pair) {
        pair.second->setUp(this); m_mcmcCallBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack<MCMC>& callback) { insertCallBack({key, &callback}); }
    virtual void removeCallBack(std::string key) {
        if ( m_mcmcCallBacks.contains(key) )
            m_mcmcCallBacks.remove(key);
        else
            throw std::logic_error("MCMC: callback of key `" + key + "` cannot be removed.");
    }
    const CallBack<MCMC>& getMCMCCallBack(std::string key){ return m_mcmcCallBacks.get(key); }

    virtual void setUp() { m_mcmcCallBacks.setUp(this); m_numSteps = m_numSweeps = 0; }
    virtual void tearDown() { m_mcmcCallBacks.tearDown(); m_numSteps = m_numSweeps = 0; }
    virtual void onSweepBegin() { m_mcmcCallBacks.onSweepBegin(); }
    virtual void onSweepEnd() { m_mcmcCallBacks.onSweepEnd(); }
    virtual void onStepBegin() { m_mcmcCallBacks.onStepBegin(); }
    virtual void onStepEnd() { m_mcmcCallBacks.onStepEnd(); }
    virtual bool doMetropolisHastingsStep() = 0;

    std::tuple<size_t, size_t> doMHSweep(size_t burn=1);
};


}

#endif
