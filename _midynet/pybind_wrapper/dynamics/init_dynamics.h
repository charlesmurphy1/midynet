#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/dynamics/binary_dynamics.h"
#include "FastMIDyNet/dynamics/python/dynamics.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initDynamicsBaseClass(py::module& m){
    py::class_<Dynamics, PyDynamics<>>(m, "Dynamics")
        .def(py::init<RandomGraph&, size_t, size_t>(),
            py::arg("random_graph"), py::arg("num_states"), py::arg("num_steps"))
        .def("get_state", &Dynamics::getState)
        .def("get_past_states", &Dynamics::getPastStates)
        .def("get_future_states", &Dynamics::getFutureStates)
        .def("set_state", &Dynamics::setState, py::arg("state"))
        .def("get_graph", &Dynamics::getGraph)
        .def("set_graph", &Dynamics::setGraph, py::arg("graph"))
        .def("get_random_graph", &Dynamics::getRandomGraph)
        .def("get_size", &Dynamics::getSize)
        .def("get_num_states", &Dynamics::getNumStates)
        .def("sample", py::overload_cast<const State&, bool>(&Dynamics::sample),
            py::arg("state"), py::arg("async")=false)
        .def("sample", py::overload_cast<bool>(&Dynamics::sample),
            py::arg("async")=false)
        .def("sample_state", py::overload_cast<const State&, bool>(&Dynamics::sampleState),
            py::arg("state"), py::arg("async")=false)
        .def("sample_state", py::overload_cast<bool>(&Dynamics::sampleState),
            py::arg("async")=false)
        .def("get_random_state", &Dynamics::getRandomState)
        .def("get_neighbors_state", &Dynamics::getNeighborsState,
            py::arg("state"))
        .def("get_vertex_neighbor_state", &Dynamics::getVertexNeighborsState,
            py::arg("idx"))
        .def("sync_update_state", &Dynamics::syncUpdateState)
        .def("async_update_state", &Dynamics::asyncUpdateState,
            py::arg("num_updates")=1)
        .def("get_log_likelihood", &Dynamics::getLogLikelihood)
        .def("get_log_prior", &Dynamics::getLogPrior)
        .def("get_log_joint", &Dynamics::getLogJoint)
        .def("get_transition_prob", &Dynamics::getTransitionProb,
            py::arg("prev_vertex_state"), py::arg("next_vertex_state"),
            py::arg("neighbor_state"))
        .def("get_transition_probs", &Dynamics::getTransitionProbs,
            py::arg("prev_vertex_state"), py::arg("neighbor_state"))
        .def("get_log_likelihood_ratio", &Dynamics::getLogLikelihoodRatio,
            py::arg("move"))
        .def("get_log_prior_ratio", &Dynamics::getLogPriorRatio,
            py::arg("move"))
        .def("get_log_joint_ratio", &Dynamics::getLogJointRatio,
            py::arg("move"))
        .def("apply_move", &Dynamics::applyMove,
            py::arg("move"))
        ;
}

void initBinaryDynamicsBaseClass(py::module& m){
    py::class_<BinaryDynamics, Dynamics, PyBinaryDynamics<>>(m, "BinaryDynamics")
        .def(py::init<RandomGraph&, size_t>(),
            py::arg("random_graph"), py::arg("num_steps"))
        .def("get_activation_prob", &BinaryDynamics::getActivationProb, py::arg("neighbor_state"))
        .def("get_deactivation_prob", &BinaryDynamics::getDeactivationProb, py::arg("neighbor_state"));
}

}
#endif
