#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"

namespace py = pybind11;


PYBIND11_MODULE(fast_midynet, m) {
    py::class_<FastMIDyNet::DoubleEdgeSwapProposer>(m, "DoubleEdgeSwap")
        .def(py::init<>());

    m.def("sample_without_replacement_uniformly", &FastMIDyNet::sampleUniformlySequenceWithoutReplacement,
            py::arg("n"), py::arg("k"));
}
