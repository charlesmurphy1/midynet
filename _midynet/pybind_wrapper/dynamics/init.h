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
        .def(py::init<size_t, double, double, double, double>(),
            py::arg("num_steps"), py::arg("nu"),
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5)
        .def(py::init<RandomGraph&, size_t, double, double, double, double>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("nu"),
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5);

    py::class_<DegreeDynamics, BinaryDynamics>(m, "DegreeDynamics")
        .def(py::init<size_t, double>(),
            py::arg("num_steps"), py::arg("C"))
        .def(py::init<RandomGraph&, size_t, double>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("C"));

    py::class_<IsingGlauberDynamics, BinaryDynamics>(m, "IsingGlauberDynamics")
        .def(py::init<size_t, double>(),
            py::arg("num_steps"), py::arg("coupling"))
        .def(py::init<RandomGraph&, size_t, double>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("coupling"));

    py::class_<SISDynamics, BinaryDynamics>(m, "SISDynamics")
        .def(py::init<size_t, double, double>(),
            py::arg("num_steps"), py::arg("infection_prob"),
            py::arg("recovery_prob")=0.5)
        .def(py::init<RandomGraph&, size_t, double, double>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("infection_prob"),
            py::arg("recovery_prob")=0.5);
}

}

#endif
