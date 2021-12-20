#ifndef FAST_MIDYNET_PYWRAPPER_INIT_GENERATOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_GENERATOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/generators.h"

void initGenerators(pybind11::module& m){

    /* Random variable generators */
    m.def("generateCategorical", &FastMIDyNet::generateCategorical<double, int>, pybind11::arg("weights"));
    m.def("generateCategorical", &FastMIDyNet::generateCategorical<int, int>, pybind11::arg("weights"));
    m.def("sampleSequenceWithoutReplacement", &FastMIDyNet::sampleUniformlySequenceWithoutReplacement, pybind11::arg("n"), pybind11::arg("k"));
    m.def("sampleRandomComposition", &FastMIDyNet::sampleRandomComposition, pybind11::arg("n"), pybind11::arg("k"));
    m.def("sampleRandomWeakComposition", &FastMIDyNet::sampleRandomWeakComposition, pybind11::arg("n"), pybind11::arg("k"));
    m.def("sampleRandomRestrictedPartition", &FastMIDyNet::sampleRandomRestrictedPartition, pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("numSteps")=0);
    m.def("sampleRandomPermutation", &FastMIDyNet::sampleRandomPermutation, pybind11::arg("nk"));

    /* Random graph generators */
    m.def("generateDCSBM", &FastMIDyNet::generateDCSBM, pybind11::arg("blocks"), pybind11::arg("edgeMatrix"), pybind11::arg("degrees"));
    m.def("generateSBM", &FastMIDyNet::generateSBM, pybind11::arg("blocks"), pybind11::arg("edgeMatrix"));
    m.def("generateCM", &FastMIDyNet::generateCM, pybind11::arg("degrees"));


}

#endif
