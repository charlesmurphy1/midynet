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
    const MCMCType* m_mcmcPtr;
public:
    void setMCMC(const MCMCType& mcmc) { m_mcmcPtr = &mcmc; }
    virtual void onBegin() { }
    virtual void onEnd() { }
    virtual void onStepBegin() { }
    virtual void onStepEnd() { }
    virtual void onSweepBegin() { }
    virtual void onSweepEnd() { }
    virtual void clear() { }

};

template <typename MCMCType>
class CallBackMap{
private:
    std::map<std::string, CallBack<MCMCType>*> m_callbacksMap;
    const MCMCType* m_mcmcPtr = nullptr;
public:
    CallBackMap() {}
    CallBackMap(const MCMCType& mcmc): m_mcmcPtr(&mcmc) {}
    CallBackMap(const CallBackMap& callBacks): m_callbacksMap(callBacks.m_callbacksMap) {}

    void reset() { for(auto c : m_callbacksMap) c.second->reset(); }
    void onBegin() { for(auto c : m_callbacksMap) c.second->onBegin(); }
    void onEnd() { for(auto c : m_callbacksMap) c.second->onEnd(); }
    void onStepBegin() { for(auto c : m_callbacksMap) c.second->onStepBegin(); }
    void onStepEnd() { for(auto c : m_callbacksMap) c.second->onStepEnd(); }
    void onSweepBegin() { for(auto c : m_callbacksMap) c.second->onSweepBegin(); }
    void onSweepEnd() { for(auto c : m_callbacksMap) c.second->onSweepEnd(); }
    void clear() { for(auto c : m_callbacksMap) c.second->clear(); }


    void setMCMC(const MCMCType& mcmc) { m_mcmcPtr = &mcmc; }
    const CallBack<MCMCType>& get(std::string key) const { return *m_callbacksMap.at(key); }
    void insert(std::pair<std::string, CallBack<MCMCType>*> pair) {
        m_callbacksMap.insert(pair);
        m_callbacksMap.at(pair.first)->setMCMC(*m_mcmcPtr);
    }
    void insert(std::string key, CallBack<MCMCType>& callback) { insert({key, &callback}); }
    void remove(std::string key) { m_callbacksMap.erase(key); }
    size_t size() const { return m_callbacksMap.size(); }
    bool contains(std::string key) { return m_callbacksMap.count(key) != 0; }
};

}

#endif
