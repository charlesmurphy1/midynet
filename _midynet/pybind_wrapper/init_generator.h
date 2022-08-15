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
    m.def("sampleMultinomial", &sampleMultinomial, py::arg("n"), py::arg("p"));
    m.def("sampleRandomNeighbor", &sampleRandomNeighbor, py::arg("graph"), py::arg("vertex"), py::arg("with_multiplicity")=true);

    /* Random graph generators */
    m.def("generateErdosRenyi", &generateErdosRenyi, py::arg("size"), py::arg("edge_count"), py::arg("with_self_loops")=true);
    m.def("generateMultiGraphErdosRenyi", &generateMultiGraphErdosRenyi, py::arg("size"), py::arg("edge_count"), py::arg("with_self_loops")=true);
    m.def("generateStubLabeledErdosRenyi", &generateStubLabeledErdosRenyi, py::arg("size"), py::arg("edge_count"), py::arg("with_self_loops")=true);
    m.def("generateCM", &generateCM, py::arg("degrees"));
    m.def("generateDCSBM", &generateDCSBM, py::arg("blocks"), py::arg("labelGraph"), py::arg("degrees"));
    m.def("generateSBM", &generateSBM, py::arg("blocks"), py::arg("labelGraph"), py::arg("with_self_loops")=true);
    m.def("generateMultiGraphSBM", &generateMultiGraphSBM, py::arg("blocks"), py::arg("labelGraph"), py::arg("with_self_loops")=true);
    m.def("generateStubLabeledSBM", &generateStubLabeledSBM, py::arg("blocks"), py::arg("labelGraph"), py::arg("with_self_loops")=true);


}

}

#endif
