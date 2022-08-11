#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/data/python/data_model.hpp"
#include "FastMIDyNet/data/dynamics/dynamics.hpp"
#include "FastMIDyNet/data/dynamics/binary_dynamics.hpp"
#include "FastMIDyNet/data/dynamics/cowan.hpp"
#include "FastMIDyNet/data/dynamics/degree.hpp"
#include "FastMIDyNet/data/dynamics/glauber.hpp"
#include "FastMIDyNet/data/dynamics/sis.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

template <typename GraphPriorType>
py::class_<Dynamics<GraphPriorType>, DataModel<GraphPriorType>, PyDynamics<GraphPriorType>> declareDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<Dynamics<GraphPriorType>, DataModel<GraphPriorType>, PyDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, size_t, bool, bool>(),
            py::arg("num_states"),
            py::arg("num_steps"),
            py::arg("normalize")=true,
            py::arg("async")=false)
        .def(py::init<GraphPriorType&, size_t, size_t, bool, bool>(),
            py::arg("graph_prior"), py::arg("num_states"),
            py::arg("num_steps"), py::arg("normalize")=true,
            py::arg("async")=false)
        .def("get_state", &Dynamics<GraphPriorType>::getState)
        .def("set_state", &Dynamics<GraphPriorType>::setState, py::arg("state"))
        .def("get_neighbors_state", &Dynamics<GraphPriorType>::getNeighborsState)
        .def("get_past_states", &Dynamics<GraphPriorType>::getPastStates)
        .def("get_past_neighbors_states", &Dynamics<GraphPriorType>::getNeighborsPastStates)
        .def("get_future_states", &Dynamics<GraphPriorType>::getFutureStates)
        .def("get_num_states", &Dynamics<GraphPriorType>::getNumStates)
        .def("get_num_steps", &Dynamics<GraphPriorType>::getNumSteps)
        .def("set_num_steps", &Dynamics<GraphPriorType>::setNumSteps)
        .def("get_random_state", &Dynamics<GraphPriorType>::getRandomState)
        .def("normalizeCoupling", &Dynamics<GraphPriorType>::normalizeCoupling)
        .def("sync_update_state", &Dynamics<GraphPriorType>::syncUpdateState)
        .def("async_update_state", &Dynamics<GraphPriorType>::asyncUpdateState,
            py::arg("num_updates")=1)
        .def("get_transition_prob", &Dynamics<GraphPriorType>::getTransitionProb,
            py::arg("prev_vertex_state"), py::arg("next_vertex_state"), py::arg("neighbor_state"))
        .def("get_transition_probs", &Dynamics<GraphPriorType>::getTransitionProbs,
            py::arg("prev_vertex_state"), py::arg("neighbor_state"))
        ;
}

template <typename GraphPriorType>
py::class_<BinaryDynamics<GraphPriorType>, Dynamics<GraphPriorType>, PyBinaryDynamics<GraphPriorType>> declareBinaryDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<BinaryDynamics<GraphPriorType>, Dynamics<GraphPriorType>, PyBinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<GraphPriorType&, size_t, double, double, bool, bool, size_t>(),
             py::arg("random_graph"), py::arg("num_steps"),
             py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.,
             py::arg("normalize")=true, py::arg("async")=false,
             py::arg("num_inital_active")=-1)
        .def(py::init<size_t, double, double, bool, bool, size_t>(),
             py::arg("num_steps"),
             py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.,
             py::arg("normalize")=true, py::arg("async")=false,
             py::arg("num_inital_active")=-1)
        .def("get_initial_active", &BinaryDynamics<GraphPriorType>::getNumInitialActive)
        .def("set_initial_active", &BinaryDynamics<GraphPriorType>::setNumInitialActive, py::arg("num_active"))
        .def("get_activation_prob", &BinaryDynamics<GraphPriorType>::getActivationProb, py::arg("neighbor_state"))
        .def("get_activation_prob", &BinaryDynamics<GraphPriorType>::getActivationProb, py::arg("neighbor_state"))
        .def("get_deactivation_prob", &BinaryDynamics<GraphPriorType>::getDeactivationProb, py::arg("neighbor_state"))
        .def("set_auto_activation_prob", &BinaryDynamics<GraphPriorType>::setAutoActivationProb, py::arg("auto_activation_prob"))
        .def("set_auto_deactivation_prob", &BinaryDynamics<GraphPriorType>::setAutoDeactivationProb, py::arg("auto_deactivation_prob"))
        .def("get_auto_activation_prob", &BinaryDynamics<GraphPriorType>::getAutoActivationProb)
        .def("get_auto_deactivation_prob", &BinaryDynamics<GraphPriorType>::getAutoDeactivationProb);
}

template<typename GraphPriorType>
py::class_<CowanDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareCowanDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<CowanDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double, double, double, double, double, double, bool, bool, size_t>(),
            py::arg("num_steps"), py::arg("nu"),
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.,
            py::arg("normalize")=true, py::arg("async")=false,
            py::arg("num_active")=1)
        .def(py::init<GraphPriorType&, size_t, double, double, double, double, double, double, bool, bool, size_t>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("nu"),
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.,
            py::arg("normalize")=true, py::arg("async")=false,
            py::arg("num_active")=1)
        .def("get_a", &CowanDynamics<GraphPriorType>::getA)
        .def("set_a", &CowanDynamics<GraphPriorType>::setA, py::arg("a"))
        .def("get_nu", &CowanDynamics<GraphPriorType>::getNu)
        .def("set_nu", &CowanDynamics<GraphPriorType>::setNu, py::arg("nu"))
        .def("get_mu", &CowanDynamics<GraphPriorType>::getMu)
        .def("set_mu", &CowanDynamics<GraphPriorType>::setMu, py::arg("mu"))
        .def("get_eta", &CowanDynamics<GraphPriorType>::getEta)
        .def("set_eta", &CowanDynamics<GraphPriorType>::setEta, py::arg("eta"));
}

template<typename GraphPriorType>
py::class_<DegreeDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareDegreeDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<DegreeDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double>(),
            py::arg("num_steps"), py::arg("C"))
        .def(py::init<GraphPriorType&, size_t, double>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("C"))
        .def("get_c", &DegreeDynamics<GraphPriorType>::getC)
        .def("set_c", &DegreeDynamics<GraphPriorType>::setC, py::arg("c"));
}

template<typename GraphPriorType>
py::class_<GlauberDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareGlauberDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<GlauberDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double, double, double, bool, bool, size_t>(),
            py::arg("num_steps"), py::arg("coupling"),
            py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.,
            py::arg("normalize")=true, py::arg("async")=false,
            py::arg("num_active")=-1)
        .def(py::init<GraphPriorType&, size_t, double, double, double, bool, bool, size_t>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("coupling"),
            py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.,
            py::arg("normalize")=true, py::arg("async")=false,
            py::arg("num_active")=-1)
        .def("get_coupling", &GlauberDynamics<GraphPriorType>::getCoupling)
        .def("set_coupling", &GlauberDynamics<GraphPriorType>::setCoupling, py::arg("coupling"));
}

template<typename GraphPriorType>
py::class_<SISDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareSISDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<SISDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double, double, double, double, bool, bool, size_t>(),
            py::arg("num_steps"), py::arg("infection_prob"), py::arg("recovery_prob")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.,
            py::arg("normalize")=true, py::arg("async")=false,
            py::arg("num_active")=1)
        .def(py::init<GraphPriorType&, size_t, double, double, double, double, bool, bool, size_t>(),
            py::arg("random_graph"), py::arg("num_steps"), py::arg("infection_prob"), py::arg("recovery_prob")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.,
            py::arg("normalize")=true, py::arg("async")=false,
            py::arg("num_active")=1)
        .def("get_infection_prob", &SISDynamics<GraphPriorType>::getInfectionProb)
        .def("set_infection_prob", &SISDynamics<GraphPriorType>::setInfectionProb, py::arg("infection_prob"))
        .def("get_recovery_prob", &SISDynamics<GraphPriorType>::getRecoveryProb)
        .def("set_recovery_prob", &SISDynamics<GraphPriorType>::setRecoveryProb, py::arg("recovery_prob"));
}



void initDynamics(py::module& m){

    declareDynamicsBaseClass<RandomGraph>(m, "Dynamics");
    declareDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDynamics");
    declareDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDynamics");

    declareBinaryDynamicsBaseClass<RandomGraph>(m, "BinaryDynamics");
    declareBinaryDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledBinaryDynamics");
    declareBinaryDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledBinaryDynamics");

    declareCowanDynamicsBaseClass<RandomGraph>(m, "CowanDynamics");
    declareCowanDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledCowanDynamics");
    declareCowanDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledCowanDynamics");

    declareDegreeDynamicsBaseClass<RandomGraph>(m, "DegreeDynamics");
    declareDegreeDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDegreeDynamics");
    declareDegreeDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDegreeDynamics");

    declareGlauberDynamicsBaseClass<RandomGraph>(m, "GlauberDynamics");
    declareGlauberDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledGlauberDynamics");
    declareGlauberDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledGlauberDynamics");

    declareSISDynamicsBaseClass<RandomGraph>(m, "SISDynamics");
    declareSISDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledSISDynamics");
    declareSISDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledSISDynamics");
}

}
#endif
