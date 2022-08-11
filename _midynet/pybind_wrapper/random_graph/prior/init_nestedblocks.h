#ifndef FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_NESTEDBLOCK_H
#define FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_NESTEDBLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "BaseGraph/types.h"

#include "FastMIDyNet/random_graph/prior/python/nested_block.hpp"
#include "FastMIDyNet/random_graph/prior/nested_block.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initNestedBlockPrior(py::module& m){
    py::class_<NestedBlockPrior, BlockPrior, PyNestedBlockPrior<>>(m, "NestedBlockPrior")
        .def(py::init<>())
        .def(py::init<size_t, NestedBlockCountPrior&>(), py::arg("size"), py::arg("nested_block_count_prior"))
        .def("get_nested_state", [](const NestedBlockPrior& self){ return self.getNestedState(); })
        .def("get_nested_state", [](const NestedBlockPrior& self, Level level){ return self.getNestedState(level); }, py::arg("level"))
        .def("set_nested_state", [](NestedBlockPrior& self, std::vector<std::vector<BlockIndex>> state){ self.setNestedState(state); })
        .def("get_depth", &NestedBlockPrior::getDepth)
        .def("get_nested_block_count_prior", &NestedBlockPrior::getNestedBlockCountPrior)
        .def("set_nested_block_count_prior", &NestedBlockPrior::setNestedBlockCountPrior, py::arg("prior"))
        .def("get_nested_block_count", [](const NestedBlockPrior& self){ return self.getNestedBlockCount(); })
        .def("get_nested_block_count", [](const NestedBlockPrior& self, Level level){ return self.getNestedBlockCount(level); })
        .def("get_nested_max_block_count", [](const NestedBlockPrior& self){ return self.getNestedMaxBlockCount(); })
        .def("get_nested_max_block_count", [](const NestedBlockPrior& self, Level level){ return self.getNestedMaxBlockCount(level); })
        .def("get_nested_effective_block_count", [](const NestedBlockPrior& self){ return self.getNestedEffectiveBlockCount(); })
        .def("get_nested_effective_block_count", [](const NestedBlockPrior& self, Level level){ return self.getNestedEffectiveBlockCount(level); })
        .def("get_nested_vertex_counts", [](const NestedBlockPrior& self){ return self.getNestedVertexCounts(); })
        .def("get_nested_vertex_counts", [](const NestedBlockPrior& self, Level level){ return self.getNestedVertexCounts(level); })
        .def("get_nested_abs_vertex_counts", [](const NestedBlockPrior& self){ return self.getNestedAbsVertexCounts(); })
        .def("get_nested_abs_vertex_counts", [](const NestedBlockPrior& self, Level level){ return self.getNestedAbsVertexCounts(level); })
        .def("get_block_of_id", [](const NestedBlockPrior& self, BaseGraph::VertexIndex vertex, Level level){ return self.getBlockOfIdx(vertex, level); })
        .def("get_nested_block_of_id", [](const NestedBlockPrior& self, BlockIndex vertex, Level level){ return self.getNestedBlockOfIdx(vertex, level); })
        .def("reduce_hierarchy", [](NestedBlockPrior&self, Level minLevel=0){ self.reduceHierarchy(minLevel);}, py::arg("min_level")=0)
        .def("sample_state", [](const NestedBlockPrior& self, Level level){ return self.sampleState(level); }, py::arg("level") )
        .def("sample_state", [](NestedBlockPrior& self){ self.sampleState(); } )
        .def("is_valid_block_move", &NestedBlockPrior::isValidBlockMove, py::arg("move") )
        .def("creating_new_block", &NestedBlockPrior::creatingNewBlock, py::arg("move") )
        .def("destroying_block", &NestedBlockPrior::destroyingBlock, py::arg("move") )
        .def("creating_new_Level", &NestedBlockPrior::creatingNewLevel, py::arg("move") )
        ;

    py::class_<NestedBlockUniformHyperPrior, NestedBlockPrior>(m, "NestedBlockUniformHyperPrior")
        .def(py::init<size_t>(), py::arg("graph_size"))
        ;
}

}

#endif
