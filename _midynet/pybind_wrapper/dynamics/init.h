#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_dynamics.h"
#include "FastMIDyNet/dynamics/cowan.h"
#include "FastMIDyNet/dynamics/degree.h"
#include "FastMIDyNet/dynamics/ising-glauber.h"
#include "FastMIDyNet/dynamics/sis.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initDynamics(py::module& m){
    initDynamicsBaseClass(m);

    initBinaryDynamicsBaseClass(m);

    py::class_<CowanDynamics, BinaryDynamics>(m, "CowanDynamics")
        .def(py::init<size_t, double, double, double, double, bool, bool>(),
            py::arg("num_steps"), py::arg("nu"),
            py::arg("a")=1, py::arg("mu")=1,
            py::arg("eta")=0.5, py::arg("normalize")=true, py::arg("cache")=false)
        .def(py::init<RandomGraph&, size_t, double, double, double, double, bool, bool>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("nu"),
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5,
            py::arg("normalize")=true, py::arg("cache")=false)
        .def("get_a", &CowanDynamics::getA)
        .def("set_a", &CowanDynamics::setA, py::arg("a"))
        .def("get_nu", &CowanDynamics::getNu)
        .def("set_nu", &CowanDynamics::setNu, py::arg("nu"))
        .def("get_mu", &CowanDynamics::getMu)
        .def("set_mu", &CowanDynamics::setMu, py::arg("mu"))
        .def("get_eta", &CowanDynamics::getEta)
        .def("set_eta", &CowanDynamics::setEta, py::arg("eta"));

    py::class_<DegreeDynamics, BinaryDynamics>(m, "DegreeDynamics")
        .def(py::init<size_t, double, bool, bool>(),
            py::arg("num_steps"), py::arg("C"), py::arg("normalize")=true, py::arg("cache")=false)
        .def(py::init<RandomGraph&, size_t, double, bool, bool>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("C"),
            py::arg("normalize")=true, py::arg("cache")=false)
        .def("get_c", &DegreeDynamics::getC)
        .def("set_c", &DegreeDynamics::setC, py::arg("c"));

    py::class_<IsingGlauberDynamics, BinaryDynamics>(m, "IsingGlauberDynamics")
        .def(py::init<size_t, double, bool, bool>(),
            py::arg("num_steps"), py::arg("coupling"), py::arg("normalize")=true, py::arg("cache")=false)
        .def(py::init<RandomGraph&, size_t, double, bool, bool>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("coupling"),
            py::arg("normalize")=true, py::arg("cache")=false)
        .def("get_coupling", &IsingGlauberDynamics::getCoupling)
        .def("set_coupling", &IsingGlauberDynamics::setCoupling, py::arg("coupling"));

    py::class_<SISDynamics, BinaryDynamics>(m, "SISDynamics")
        .def(py::init<size_t, double, double, double, bool, bool>(),
            py::arg("num_steps"), py::arg("infection_prob"),
            py::arg("recovery_prob")=0.5, py::arg("auto_infection_prob")=1e-6,
            py::arg("normalize")=true, py::arg("cache")=false)
        .def(py::init<RandomGraph&, size_t, double, double, double, bool, bool>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("infection_prob"),
            py::arg("recovery_prob")=0.5, py::arg("auto_infection_prob")=1e-6,
            py::arg("normalize")=true, py::arg("cache")=false)
        .def("get_infection_prob", &SISDynamics::getInfectionProb)
        .def("set_infection_prob", &SISDynamics::setInfectionProb, py::arg("infection_prob"))
        .def("get_recovery_prob", &SISDynamics::getRecoveryProb)
        .def("set_recovery_prob", &SISDynamics::setRecoveryProb, py::arg("recovery_prob"))
        .def("get_auto_infection_prob", &SISDynamics::getAutoInfectionProb)
        .def("set_auto_infection_prob", &SISDynamics::setAutoInfectionProb, py::arg("auto_infection_prob"));
}

}

#endif
