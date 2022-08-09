#ifndef FAST_MIDYNET_PYTHON_CALLBACK_HPP
#define FAST_MIDYNET_PYTHON_CALLBACK_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "FastMIDyNet/mcmc/callbacks/collector.hpp"
#include "FastMIDyNet/mcmc/callbacks/verbose.h"

namespace FastMIDyNet{

/* CallBack  base class */
template<typename MCMCType, typename BaseClass = CallBack<MCMCType>>
class PyCallBack: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */

    /* Abstract methods */
    void onBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onBegin, ); }
    void onEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onEnd, ); }
    void onStepBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onStepBegin, ); }
    void onStepEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onStepEnd, ); }
    void onSweepBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onSweepBegin, ); }
    void onSweepEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onSweepEnd, ); }
    void clear() override { PYBIND11_OVERRIDE(void, BaseClass, clear, ); }
};

/* Verbose classes */
template<typename BaseClass = Verbose>
class PyVerbose: public PyCallBack<MCMC, BaseClass>{
public:
    using PyCallBack<MCMC, BaseClass>::PyCallBack;
    /* Pure abstract methods */
    std::string getMessage() const override { PYBIND11_OVERRIDE_PURE(std::string, BaseClass, getMessage, ); }

    /* Abstract methods */

};

template<typename BaseClass = VerboseDisplay>
class PyVerboseDisplay: public PyCallBack<MCMC, BaseClass>{
public:
    using PyCallBack<MCMC, BaseClass>::PyCallBack;
    /* Pure abstract methods */
    void writeMessage(std::string message) override {PYBIND11_OVERRIDE_PURE(void, BaseClass, writeMessage, message); }

    /* Abstract methods */
    void onBegin() override {PYBIND11_OVERRIDE(void, BaseClass, onBegin, ); }
    void onEnd() override {PYBIND11_OVERRIDE(void, BaseClass, onEnd, ); }

};

template<typename BaseClass = LogJointRatioVerbose>
class PyLogJointRatioVerbose: public PyVerbose<BaseClass>{
public:
    using PyVerbose<BaseClass>::PyVerbose;
    /* Pure abstract methods */
    double updateSaved() const override { PYBIND11_OVERRIDE_PURE(double, BaseClass, updateSaved, ); }

    /* Abstract methods */
    std::string getMessage() const override { PYBIND11_OVERRIDE(std::string, BaseClass, getMessage, ); }

};

/* Collector classes */
template<typename MCMCType, typename BaseClass = Collector<MCMCType>>
class PyCollector: public PyCallBack<MCMCType, BaseClass>{
public:
    using PyCallBack<MCMCType, BaseClass>::PyCallBack;
    /* Pure abstract methods */
    void collect() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, collect, ); }
    void clear() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, clear, ); }

    /* Abstract methods */

};


}

#endif
