#ifndef FAST_MIDYNET_CALLBACK_H
#define FAST_MIDYNET_CALLBACK_H

#include <cstddef>
#include <vector>
#include <map>

namespace FastMIDyNet{

// class MCMC;

template <typename MCMCType>
class CallBack{
protected:
    MCMCType* m_mcmcPtr;
public:
    virtual void setUp(MCMCType* mcmcPtr) { m_mcmcPtr = mcmcPtr; }
    virtual void tearDown() { }
    virtual void onBegin() { };
    virtual void onEnd() { };
    virtual void onStepBegin() { };
    virtual void onStepEnd() { };
    virtual void onSweepBegin() { };
    virtual void onSweepEnd() { };

};

template <typename MCMCType>
class CallBackMap{
private:
    std::map<std::string, CallBack<MCMCType>*> m_callbacksMap;
public:
    CallBackMap() {}
    CallBackMap(std::map<std::string, CallBack<MCMCType>*> callBacks): m_callbacksMap(callBacks) {}
    CallBackMap(const CallBackMap& callBacks): m_callbacksMap(callBacks.m_callbacksMap) {}

    void setUp(MCMCType* mcmcPtr) { for(auto c : m_callbacksMap) c.second->setUp(mcmcPtr); }
    void tearDown() { for(auto c : m_callbacksMap) c.second->tearDown(); }
    void onBegin() { for(auto c : m_callbacksMap) c.second->onBegin(); }
    void onEnd() { for(auto c : m_callbacksMap) c.second->onEnd(); }
    void onStepBegin() { for(auto c : m_callbacksMap) c.second->onStepBegin(); }
    void onStepEnd() { for(auto c : m_callbacksMap) c.second->onStepEnd(); }
    void onSweepBegin() { for(auto c : m_callbacksMap) c.second->onSweepBegin(); }
    void onSweepEnd() { for(auto c : m_callbacksMap) c.second->onSweepEnd(); }


    const CallBack<MCMCType>& get(std::string key) const { return *m_callbacksMap.at(key); }
    void insert(std::pair<std::string, CallBack<MCMCType>*> pair) { m_callbacksMap.insert(pair); }
    void insert(std::string key, CallBack<MCMCType>& callback) { insert({key, &callback}); }
    void remove(std::string key) { m_callbacksMap.erase(key); }
    size_t size() const { return m_callbacksMap.size(); }
    bool contains(std::string key) { return m_callbacksMap.count(key) != 0; }
};

}

#endif
