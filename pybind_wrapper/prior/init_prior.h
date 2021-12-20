#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PRIOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PRIOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "sbm/init_sbmpriors.h"


// template <typename StateType>
// class PyPrior: public FastMIDyNet::Prior<StateType>{
//     public:
//         virtual void sampleState() { PYBIND11_OVERLOAD_PURE(void, FastMIDyNet::Prior<StateType>, sampleState); }
//         virtual void samplePriors()  { PYBIND11_OVERLOAD_PURE(void, FastMIDyNet::Prior<StateType>, samplePriors); }
//         virtual double getLogLikelihood() const { PYBIND11_OVERLOAD_PURE(double, FastMIDyNet::Prior<StateType>, getLogLikelihood); }
//         virtual double getLogPrior() { PYBIND11_OVERLOAD_PURE(double, FastMIDyNet::Prior<StateType>, getLogPrior); }
//         virtual void checkSelfConsistency() const { PYBIND11_OVERLOAD_PURE(void, FastMIDyNet::Prior<StateType>, checkSelfConsistency); };
// };

// template<typename StateType>
// void definePriorBaseClass(pybind11::module& m, std::string pyName){
//     pybind11::class_<FastMIDyNet::Prior<StateType>>(m, pyName.c_str())
//         .def(pybind11::init<>())
//         .def("get_state", &FastMIDyNet::Prior<StateType>::getState)
//         .def("set_state", &FastMIDyNet::Prior<StateType>::setState)
//         .def("sample", &FastMIDyNet::Prior<StateType>::sample)
//         // .def("sample_state", &FastMIDyNet::Prior<StateType>::sampleState)
//         // .def("sample_priors", &FastMIDyNet::Prior<StateType>::samplePriors)
//         // .def("get_logLikelihood", &FastMIDyNet::Prior<StateType>::getLogLikelihood)
//         // .def("get_logPrior", &FastMIDyNet::Prior<StateType>::getLogPrior)
//         .def("get_logJoint", &FastMIDyNet::Prior<StateType>::getLogJoint);
// }

void initPrior(pybind11::module& m){
    // pybind11::module mBase = m.def_submodule("_base");
    // definePriorBaseClass<size_t>(mBase, "PriorUnsignedInt");

    pybind11::module mSBM = m.def_submodule("_sbm");
    initSBMPrior(mSBM);
}

#endif
