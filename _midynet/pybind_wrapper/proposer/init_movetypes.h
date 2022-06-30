#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/proposer/movetypes.h"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label>
py::class_<LabelMove<Label>> declareLabelMove(py::module& m, std::string pyName){
    return py::class_<LabelMove<Label>>(m, pyName.c_str())
        .def(py::init<BaseGraph::VertexIndex, Label, Label, int>(),
            py::arg("vertex_index"), py::arg("prev_label"), py::arg("next_label"), py::arg("added_labels"))
        .def_readonly("vertex_id", &BlockMove::vertexIndex)
        .def_readonly("prev_label", &BlockMove::prevLabel)
        .def_readonly("next_label", &BlockMove::nextLabel)
        .def_readonly("added_labels", &BlockMove::addedLabels)
        .def("__repr__", [](const LabelMove<Label>& self){ return self.display(); })
        ;
}

void initMoveTypes(py::module& m){
    py::class_<GraphMove>(m, "GraphMove")
        .def(py::init<std::vector<BaseGraph::Edge>, std::vector<BaseGraph::Edge>>(),
            py::arg("removed_edges"), py::arg("added_edges"))
        .def_readonly("removed_edges", &GraphMove::removedEdges)
        .def_readonly("added_edges", &GraphMove::addedEdges)
        .def("__repr__", [](const GraphMove& self){ return self.display(); })
        ;

    declareLabelMove<BlockIndex>(m, "BlockMove");

}

}

#endif
