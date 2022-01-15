#ifndef FAST_MIDYNET_CALLBACK_H
#define FAST_MIDYNET_CALLBACK_H

#include <vector>

namespace FastMIDyNet{

class MCMC;

class CallBack{
protected:
    MCMC* m_mcmcPtr;
public:
    virtual void setUp(MCMC* mcmcPtr) { m_mcmcPtr = mcmcPtr; }
    virtual void tearDown() { }
    virtual void onBegin() { };
    virtual void onEnd() { };
    virtual void onStepBegin() { };
    virtual void onStepEnd() { };
    virtual void onSweepBegin() { };
    virtual void onSweepEnd() { };

};

class CallBackList{
private:
    std::vector<CallBack*> m_callbacksVec;
public:
    CallBackList(std::vector<CallBack*> callBacks={}):m_callbacksVec(callBacks) {}
    CallBackList(const CallBackList& callBacks): m_callbacksVec(callBacks.m_callbacksVec) {}

    void setUp(MCMC* mcmcPtr) { for(auto c : m_callbacksVec) c->setUp(mcmcPtr); }
    void tearDown() { for(auto c : m_callbacksVec) c->tearDown(); }
    void onBegin() { for(auto c : m_callbacksVec) c->onBegin(); }
    void onEnd() { for(auto c : m_callbacksVec) c->onEnd(); }
    void onStepBegin() { for(auto c : m_callbacksVec) c->onStepBegin(); }
    void onStepEnd() { for(auto c : m_callbacksVec) c->onStepEnd(); }
    void onSweepBegin() { for(auto c : m_callbacksVec) c->onSweepBegin(); }
    void onSweepEnd() { for(auto c : m_callbacksVec) c->onSweepEnd(); }

    void pushBack(CallBack& callback) { m_callbacksVec.push_back(&callback); }
    void remove(size_t idx) { m_callbacksVec.erase(m_callbacksVec.begin() + idx); }
    void popBack() { m_callbacksVec.pop_back(); }

};

}

#endif
