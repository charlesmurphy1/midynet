#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DISTANCE_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DISTANCE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/utility/distance.h"
#include "FastMIDyNet/utility/python/distance.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initDistances(py::module& m){
    py::class_< GraphDistance, PyGraphDistance<> >(m, "GraphDistance")
        .def(py::init<>())
        .def("compute", &GraphDistance::compute, py::arg("g1"), py::arg("g2"))
        ;
    py::class_< HammingDistance, GraphDistance >(m, "HammingDistance")
        .def(py::init<>())
        ;
}

}

#endif
