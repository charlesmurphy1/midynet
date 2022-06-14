#ifndef FAST_MIDYNET_MCMC_H
#define FAST_MIDYNET_MCMC_H

#include <sstream>
#include <tuple>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

class MCMC: public NestedRandomVariable{
protected:
    CallBackMap m_callBacks;
    size_t m_numSteps;
    size_t m_numSweeps;
    mutable double m_lastLogJointRatio;
    mutable double m_lastLogAcceptance;
    mutable bool m_isLastAccepted;
public:
    MCMC():
        m_numSteps(0),
        m_numSweeps(0),
        m_lastLogJointRatio(0),
        m_isLastAccepted(false) {}
    MCMC(const CallBackMap& callbacks):
        m_callBacks(callbacks),
        m_numSteps(0),
        m_numSweeps(0),
        m_lastLogJointRatio(0),
        m_isLastAccepted(false) {}

    const double getLastLogJointRatio() const { return m_lastLogJointRatio; }
    const double getLastLogAcceptance() const { return m_lastLogAcceptance; }
    const bool isLastAccepted() const { return m_isLastAccepted; }
    const size_t getNumSteps() const { return m_numSteps; }
    const size_t getNumSweeps() const { return m_numSweeps; }

    virtual const MultiGraph& getGraph() const = 0 ;
    virtual const std::vector<BlockIndex>& getBlocks() const = 0 ;
    size_t getSize() const { return getGraph().getSize(); }
    virtual const double getLogLikelihood() const = 0 ;
    virtual const double getLogPrior() const = 0 ;
    virtual const double getLogJoint() const = 0 ;

    void insertCallBack(std::pair<std::string, CallBack*> pair) {
        pair.second->setUp(this); m_callBacks.insert(pair);
    }
    void insertCallBack(std::string key, CallBack& callback) { insertCallBack({key, &callback}); }
    void removeCallBack(std::string key) { m_callBacks.remove(key); }

    virtual void setUp() { m_callBacks.setUp(this); m_numSteps = m_numSweeps = 0; }
    virtual void tearDown() { m_callBacks.tearDown(); m_numSteps = m_numSweeps = 0; }
    virtual bool _doMetropolisHastingsStep() = 0;
    bool doMetropolisHastingsStep() {
        return processRecursiveFunction<bool>([&](){ return _doMetropolisHastingsStep(); }, false);
    };

    std::pair<size_t, size_t> doMHSweep(size_t burn=1, bool checkingConsistency=false, bool checkingSafety=false);
};

}

#endif
