#ifndef FAST_MIDYNET_RV_HPP
#define FAST_MIDYNET_RV_HPP

#include <functional>

namespace FastMIDyNet{

class NestedRandomVariable{
protected:
    mutable bool m_isRoot = true;
    mutable bool m_isProcessed = false;

    template<typename RETURN_TYPE>
    RETURN_TYPE processRecursiveConstFunction(const std::function<RETURN_TYPE()>& func, RETURN_TYPE init) const {
        RETURN_TYPE ret = init;
        if (!m_isProcessed)
            ret = func();

        if ( m_isRoot ) computationFinished();
        else m_isProcessed=true;
        return ret;
    }
    void processRecursiveConstFunction(const std::function<void()>& func) const {
        if (!m_isProcessed)
            func();
        m_isProcessed=true;
        if ( m_isRoot ) computationFinished();
        else m_isProcessed=true;
    }

    template<typename RETURN_TYPE>
    RETURN_TYPE processRecursiveFunction(const std::function<RETURN_TYPE()>& func, RETURN_TYPE init) const {
        RETURN_TYPE ret = init;
        if (!m_isProcessed)
            ret = func();

        if ( m_isRoot ) computationFinished();
        else m_isProcessed=true;
        return ret;
    }
    void processRecursiveFunction(const std::function<void()>& func) {
        if (!m_isProcessed)
            func();
        m_isProcessed=true;
        if ( m_isRoot ) computationFinished();
        else m_isProcessed=true;
    }
public:
    const bool isRoot() const { return m_isRoot; }
    virtual void isRoot(bool condition) const { m_isRoot = condition; }
    void checkSelfConsistency() const { processRecursiveConstFunction([&](){ _checkSelfConsistency(); }); }
    void checkSafety() const  { processRecursiveConstFunction([&](){ _checkSafety(); }); }
    virtual void computationFinished() const { m_isProcessed = false; }

    virtual void _checkSelfConsistency() const { }
    virtual void _checkSafety() const  { }
};

}

#endif
