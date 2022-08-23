#ifndef FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_BLOCKCOUNT_H
#define FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/prior/prior.hpp"
#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/python/blockcount.hpp"

namespace py = pybind11;
namespace FastMIDyNet{


void initBlockCountPrior(py::module& m){
    py::class_<BlockCountPrior, BlockLabeledPrior<size_t>, PyBlockCountPrior<>>(m, "BlockCountPrior");

    py::class_<BlockCountDeltaPrior, BlockCountPrior>(m, "BlockCountDeltaPrior")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("block_count"));

    py::class_<BlockCountPoissonPrior, BlockCountPrior>(m, "BlockCountPoissonPrior")
        .def(py::init<>())
        .def(py::init<double>(), py::arg("mean"))
        .def("get_mean", &BlockCountPoissonPrior::getMean)
        .def("set_mean", &BlockCountPoissonPrior::setMean, py::arg("mean"))
        ;
    py::class_<BlockCountUniformPrior, BlockCountPrior>(m, "BlockCountUniformPrior")
        .def(py::init<>())
        .def(py::init<size_t, size_t>(), py::arg("min"), py::arg("max"))
        .def("get_min", &BlockCountUniformPrior::getMin)
        .def("get_max", &BlockCountUniformPrior::getMax)
        .def("set_min", &BlockCountUniformPrior::setMin, py::arg("min"))
        .def("set_max", &BlockCountUniformPrior::setMax, py::arg("max"))
        .def("set_min_max", &BlockCountUniformPrior::setMinMax, py::arg("min"), py::arg("max"))
        ;

    py::class_<NestedBlockCountPrior, BlockCountPrior, PyNestedBlockCountPrior<>>(m, "NestedBlockCountPrior")
        .def(py::init<>())
        .def("get_depth", &NestedBlockCountPrior::getDepth)
        .def("get_nested_state", [](const NestedBlockCountPrior& self){ return self.getNestedState(); })
        .def("get_nested_state", [](const NestedBlockCountPrior& self, Level level){ return self.getNestedState(level); }, py::arg("level"))
        .def("create_new_level", &NestedBlockCountPrior::createNewLevel)
        .def("destroy_last_level", &NestedBlockCountPrior::destroyLastLevel)
        .def("set_nested_state", [](NestedBlockCountPrior& self, std::vector<size_t> state){ self.setNestedState(state); }, py::arg("nested_state"))
        .def("set_nested_state", [](NestedBlockCountPrior& self, size_t state, Level level){ self.setNestedState(state, level); }, py::arg("state"), py::arg("level"))
        .def("set_nested_state_from_nested_partition", &NestedBlockCountPrior::setNestedStateFromNestedPartition, py::arg("nested_partition"))
        ;

    py::class_<NestedBlockCountUniformPrior, NestedBlockCountPrior>(m, "NestedBlockCountUniformPrior")
        .def(py::init<size_t>(), py::arg("graph_size"))
        .def("get_graph_size", &NestedBlockCountUniformPrior::getGraphSize)
        .def("set_graph_size", &NestedBlockCountUniformPrior::setGraphSize, py::arg("graph_size"))
        ;
}

}

#endif
