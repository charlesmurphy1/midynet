#ifndef FAST_MIDYNET_PYWRAPPER_INIT_MCMCBASECLASS_H
#define AST_MIDYNET_PYWRAPPER_INIT_MCMCBASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/community.hpp"
#include "FastMIDyNet/mcmc/reconstruction.hpp"
#include "FastMIDyNet/mcmc/python/mcmc.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

py::class_<MCMC, NestedRandomVariable, PyMCMC<>> declareMCMCBaseClass(py::module& m){
    return py::class_<MCMC, NestedRandomVariable, PyMCMC<>>(m, "MCMC")
        .def(py::init<double, double>(), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("get_last_log_joint_ratio", &MCMC::getLastLogJointRatio)
        .def("get_last_log_acceptance", &MCMC::getLastLogAcceptance)
        .def("is_last_accepted", &MCMC::isLastAccepted)
        .def("get_num_steps", &MCMC::getNumSteps)
        .def("get_num_sweeps", &MCMC::getNumSweeps)
        .def("get_beta_prior", &MCMC::getBetaPrior)
        .def("set_beta_prior", &MCMC::setBetaPrior, py::arg("beta"))
        .def("get_beta_likelihood", &MCMC::getBetaLikelihood)
        .def("set_beta_likelihood", &MCMC::setBetaLikelihood, py::arg("beta"))
        .def("get_log_likelihood", &MCMC::getLogLikelihood)
        .def("get_log_prior", &MCMC::getLogPrior)
        .def("get_log_joint", &MCMC::getLogJoint)
        .def("insert_callback", [](MCMC& self, std::string key, CallBack<MCMC>& callback){
            self.insertCallBack(key, callback);
        }, py::arg("key"), py::arg("callback"))
        .def("remove_callback", &MCMC::removeCallBack, py::arg("key"))
        .def("get_mcmc_callback", &MCMC::getMCMCCallBack, py::arg("key"))
        .def("sample", &MCMC::sample)
        .def("sample_prior", &MCMC::samplePrior)
        .def("on_begin", &MCMC::onBegin)
        .def("on_end", &MCMC::onEnd)
        .def("on_step_begin", &MCMC::onStepBegin)
        .def("on_step_end", &MCMC::onStepEnd)
        .def("on_sweep_begin", &MCMC::onSweepBegin)
        .def("on_sweep_end", &MCMC::onSweepEnd)
        .def("do_metropolis_hastings_step", &MCMC::doMetropolisHastingsStep)
        .def("do_MH_sweep", &MCMC::doMHSweep, py::arg("burn")=1)
        ;
}

template<typename Label>
py::class_<VertexLabelReconstructionMCMC<Label>, MCMC> declareVertexLabelReconstructionMCMCClass(py::module& m, std::string pyName){
    return py::class_<VertexLabelReconstructionMCMC<Label>, MCMC>(m, pyName.c_str())
        .def(py::init<VertexLabeledRandomGraph<Label>&, double, double>(),
            py::arg("graph_prior"), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double>(), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("set_graph_prior", &VertexLabelReconstructionMCMC<Label>::setGraphPrior, py::arg("graph_prior"))
        .def("get_graph_prior", &VertexLabelReconstructionMCMC<Label>::getGraphPrior)
        .def("get_graph", &VertexLabelReconstructionMCMC<Label>::getGraph)
        .def("set_graph", &VertexLabelReconstructionMCMC<Label>::setGraph, py::arg("graph"))
        .def("get_labels", &VertexLabelReconstructionMCMC<Label>::getLabels)
        .def("set_labels", &VertexLabelReconstructionMCMC<Label>::setLabels, py::arg("labels"), py::arg("reduce")=false)
        .def("insert_callback", [](VertexLabelReconstructionMCMC<Label>& self, std::string key, CallBack<MCMC>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("insert_callback", [](VertexLabelReconstructionMCMC<Label>& self, std::string key, CallBack<VertexLabelReconstructionMCMC<Label>>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("get_label_callback", &VertexLabelReconstructionMCMC<Label>::getLabelCallBack, py::arg("key"))
        // .def("get_log_acceptance_prob_from_label_move", &VertexLabelReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove, py::arg("move"))
        ;
}

template<typename Label>
py::class_<NestedVertexLabelReconstructionMCMC<Label>, MCMC> declareNestedVertexLabelReconstructionMCMCClass(py::module& m, std::string pyName){
    return py::class_<NestedVertexLabelReconstructionMCMC<Label>, MCMC>(m, pyName.c_str())
        .def(py::init<NestedVertexLabeledRandomGraph<Label>&, double, double>(),
            py::arg("graph_prior"), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double>(), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("set_graph_prior", &NestedVertexLabelReconstructionMCMC<Label>::setGraphPrior, py::arg("graph_prior"))
        .def("get_graph_prior", &NestedVertexLabelReconstructionMCMC<Label>::getGraphPrior)
        .def("get_graph", &NestedVertexLabelReconstructionMCMC<Label>::getGraph)
        .def("set_graph", &NestedVertexLabelReconstructionMCMC<Label>::setGraph, py::arg("graph"))
        .def("get_labels", &NestedVertexLabelReconstructionMCMC<Label>::getLabels)
        .def("get_nested_labels", &NestedVertexLabelReconstructionMCMC<Label>::getNestedLabels)
        .def("set_nested_labels", &NestedVertexLabelReconstructionMCMC<Label>::setNestedLabels, py::arg("labels"), py::arg("reduce")=false)
        .def("insert_callback", [](NestedVertexLabelReconstructionMCMC<Label>& self, std::string key, CallBack<MCMC>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("insert_callback", [](NestedVertexLabelReconstructionMCMC<Label>& self, std::string key, CallBack<NestedVertexLabelReconstructionMCMC<Label>>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("get_label_callback", &NestedVertexLabelReconstructionMCMC<Label>::getLabelCallBack, py::arg("key"))
        // .def("get_log_acceptance_prob_from_label_move", &NestedVertexLabelReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove, py::arg("move"))
        ;
}

template<typename GraphPrior>
py::class_<GraphReconstructionMCMC<GraphPrior>, MCMC> declareGraphReconstructionClass(py::module& m, std::string pyName){
    return py::class_<GraphReconstructionMCMC<GraphPrior>, MCMC>(m, pyName.c_str())
        .def(py::init<Dynamics<GraphPrior>&, double, double>(),
            py::arg("dynamics"), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double>(), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("set_data_model", &GraphReconstructionMCMC<GraphPrior>::setDataModel, py::arg("data_model"))
        .def("get_data_model", &GraphReconstructionMCMC<GraphPrior>::getDataModel)
        .def("get_graph_prior", &GraphReconstructionMCMC<GraphPrior>::getGraphPrior)
        .def("get_graph", &GraphReconstructionMCMC<GraphPrior>::getGraph)
        .def("set_graph", &GraphReconstructionMCMC<GraphPrior>::setGraph, py::arg("graph"))
        .def("insert_callback", [](GraphReconstructionMCMC<GraphPrior>& self, std::string key, CallBack<MCMC>& callback){
            self.insertCallBack(key, callback);
        }, py::arg("key"), py::arg("callback"))
        .def("insert_callback", [](GraphReconstructionMCMC<GraphPrior>& self, std::string key, CallBack<GraphReconstructionMCMC<GraphPrior>>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("get_graph_callback", &GraphReconstructionMCMC<GraphPrior>::getGraphCallBack, py::arg("key"))
        // .def("get_log_acceptance_prob_from_graph_move", &GraphReconstructionMCMC<GraphPrior>::getLogAcceptanceProbFromGraphMove, py::arg("move"))
        ;
}

template<typename Label>
py::class_<VertexLabeledGraphReconstructionMCMC<Label>, GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>> declareVertexLabeledGraphReconstructionClass(py::module& m, std::string pyName){
    return py::class_<VertexLabeledGraphReconstructionMCMC<Label>, GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>>(m, pyName.c_str())
        .def(py::init<Dynamics<VertexLabeledRandomGraph<Label>>&, double, double, double>(),
            py::arg("dynamics"), py::arg("sample_label_prob")=0.5, py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double, double>(), py::arg("sample_label_prob")=0.5, py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("get_labels", &VertexLabeledGraphReconstructionMCMC<Label>::getLabels)
        .def("set_labels", &VertexLabeledGraphReconstructionMCMC<Label>::setLabels, py::arg("labels"), py::arg("reduce")=false)
        // .def("get_log_acceptance_prob_from_label_move", &VertexLabeledGraphReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove, py::arg("move"))
        ;
}

template<typename Label>
py::class_<NestedVertexLabeledGraphReconstructionMCMC<Label>, GraphReconstructionMCMC<NestedVertexLabeledRandomGraph<Label>>> declareNestedVertexLabeledGraphReconstructionClass(py::module& m, std::string pyName){
    return py::class_<NestedVertexLabeledGraphReconstructionMCMC<Label>, GraphReconstructionMCMC<NestedVertexLabeledRandomGraph<Label>>>(m, pyName.c_str())
        .def(py::init<Dynamics<NestedVertexLabeledRandomGraph<Label>>&, double, double, double>(),
            py::arg("dynamics"), py::arg("sample_label_prob")=0.5, py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double, double>(), py::arg("sample_label_prob")=0.5, py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("get_labels", &NestedVertexLabeledGraphReconstructionMCMC<Label>::getLabels)
        .def("get_nested_labels", &NestedVertexLabeledGraphReconstructionMCMC<Label>::getNestedLabels)
        .def("set_nested_labels", &NestedVertexLabeledGraphReconstructionMCMC<Label>::setNestedLabels, py::arg("labels"), py::arg("reduce")=false)
        // .def("get_log_acceptance_prob_from_label_move", &NestedVertexLabeledGraphReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove, py::arg("move"))
        ;
}

}

#endif
