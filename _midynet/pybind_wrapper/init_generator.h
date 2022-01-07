#ifndef FAST_MIDYNET_PYWRAPPER_INIT_GENERATOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_GENERATOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/generators.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initGenerators(py::module& m){

    /* Random variable generators */
    m.def("generateCategorical", &generateCategorical<double, int>, py::arg("weights"));
    m.def("generateCategorical", &generateCategorical<int, int>, py::arg("weights"));
    m.def("sampleSequenceWithoutReplacement", &sampleUniformlySequenceWithoutReplacement, py::arg("n"), py::arg("k"));
    m.def("sampleRandomComposition", &sampleRandomComposition, py::arg("n"), py::arg("k"));
    m.def("sampleRandomWeakComposition", &sampleRandomWeakComposition, py::arg("n"), py::arg("k"));
    m.def("sampleRandomRestrictedPartition", &sampleRandomRestrictedPartition, py::arg("n"), py::arg("k"), py::arg("numSteps")=0);
    m.def("sampleRandomPermutation", &sampleRandomPermutation, py::arg("nk"));

    /* Random graph generators */
    m.def("generateDCSBM", &generateDCSBM, py::arg("blocks"), py::arg("edgeMatrix"), py::arg("degrees"));
    m.def("generateSBM", &generateSBM, py::arg("blocks"), py::arg("edgeMatrix"));
    m.def("generateCM", &generateCM, py::arg("degrees"));


}

}

#endif
