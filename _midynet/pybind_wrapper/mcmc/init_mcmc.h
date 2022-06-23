#ifndef FAST_MIDYNET_PYWRAPPER_INIT_MCMCBASECLASS_H
#define AST_MIDYNET_PYWRAPPER_INIT_MCMCBASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/community.hpp"
#include "FastMIDyNet/mcmc/reconstruction.hpp"
#include "FastMIDyNet/mcmc/python/mcmc.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/proposer/edge/edge_proposer.h"

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
        .def("samplePrior", &MCMC::samplePrior)
        .def("set_up", &MCMC::setUp)
        .def("tear_down", &MCMC::tearDown)
        .def("on_step_begin", &MCMC::onStepBegin)
        .def("on_step_end", &MCMC::onStepEnd)
        .def("on_sweep_begin", &MCMC::onSweepBegin)
        .def("on_sweep_end", &MCMC::onSweepEnd)
        .def("do_metropolis_hastings_step", &MCMC::doMetropolisHastingsStep)
        .def("do_MH_sweep", &MCMC::doMHSweep, py::arg("burn")=1)
        ;
}

template<typename Label>
py::class_<VertexLabelMCMC<Label>, MCMC> declareVertexLabelMCMCClass(py::module& m, std::string pyName){
    return py::class_<VertexLabelMCMC<Label>, MCMC>(m, pyName.c_str())
        .def(py::init<VertexLabeledRandomGraph<Label>&, LabelProposer<Label>&, double, double>(),
            py::arg("graph_prior"), py::arg("label_proposer"), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double>(), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("set_graph_prior", &VertexLabelMCMC<Label>::setGraphPrior, py::arg("graph_prior"))
        .def("get_graph_prior", &VertexLabelMCMC<Label>::getGraphPrior)
        .def("set_label_proposer", &VertexLabelMCMC<Label>::setLabelProposer, py::arg("label_proposer"))
        .def("get_label_proposer", &VertexLabelMCMC<Label>::getLabelProposer)
        .def("get_vertex_labels", &VertexLabelMCMC<Label>::getVertexLabels)
        .def("insert_callback", [](VertexLabelMCMC<Label>& self, std::string key, CallBack<MCMC>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("insert_callback", [](VertexLabelMCMC<Label>& self, std::string key, CallBack<VertexLabelMCMC<Label>>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("get_label_callback", &VertexLabelMCMC<Label>::getLabelCallBack, py::arg("key"))
        .def("get_log_acceptance_prob_from_label_move", &VertexLabelMCMC<Label>::getLogAcceptanceProbFromLabelMove, py::arg("move"))
        .def("apply_label_move", &VertexLabelMCMC<Label>::applyLabelMove, py::arg("move"))
        ;
}

template<typename GraphPrior>
py::class_<GraphReconstructionMCMC<GraphPrior>, MCMC> declareGraphReconstructionClass(py::module& m, std::string pyName){
    return py::class_<GraphReconstructionMCMC<GraphPrior>, MCMC>(m, pyName.c_str())
        .def(py::init<Dynamics<GraphPrior>&, EdgeProposer&, double, double>(),
            py::arg("dynamics"), py::arg("edge_proposer"), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double>(), py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("set_dynamics", &GraphReconstructionMCMC<GraphPrior>::setDynamics, py::arg("dynamics"))
        .def("get_dynamics", &GraphReconstructionMCMC<GraphPrior>::getDynamics)
        .def("get_graph_prior", &GraphReconstructionMCMC<GraphPrior>::getGraphPrior)
        .def("set_edge_proposer", &GraphReconstructionMCMC<GraphPrior>::setEdgeProposer, py::arg("edge_proposer"))
        .def("get_edge_proposer", &GraphReconstructionMCMC<GraphPrior>::getEdgeProposer)
        .def("get_graph", &GraphReconstructionMCMC<GraphPrior>::getGraph)
        .def("set_graph", &GraphReconstructionMCMC<GraphPrior>::setGraph)
        .def("insert_callback", [](GraphReconstructionMCMC<GraphPrior>& self, std::string key, CallBack<MCMC>& callback){
            self.insertCallBack(key, callback);
        }, py::arg("key"), py::arg("callback"))
        .def("insert_callback", [](GraphReconstructionMCMC<GraphPrior>& self, std::string key, CallBack<GraphReconstructionMCMC<GraphPrior>>& callback){
            self.insertCallBack(key, callback); }, py::arg("key"), py::arg("callback"))
        .def("get_graph_callback", &GraphReconstructionMCMC<GraphPrior>::getGraphCallBack, py::arg("key"))
        .def("get_log_acceptance_prob_from_graph_move", &GraphReconstructionMCMC<GraphPrior>::getLogAcceptanceProbFromGraphMove, py::arg("move"))
        .def("apply_graph_move", &GraphReconstructionMCMC<GraphPrior>::applyGraphMove, py::arg("move"))
        ;
}

template<typename Label>
py::class_<VertexLabeledGraphReconstructionMCMC<Label>, GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>> declareVertexLabeledGraphReconstructionClass(py::module& m, std::string pyName){
    return py::class_<VertexLabeledGraphReconstructionMCMC<Label>, GraphReconstructionMCMC<VertexLabeledRandomGraph<Label>>>(m, pyName.c_str())
        .def(py::init<Dynamics<VertexLabeledRandomGraph<Label>>&, EdgeProposer&, LabelProposer<Label>&, double, double, double>(),
            py::arg("dynamics"), py::arg("edge_proposer"), py::arg("label_proposer"), py::arg("sample_label_prob")=0.5, py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def(py::init<double, double, double>(), py::arg("sample_label_prob")=0.5, py::arg("beta_prior")=1, py::arg("beta_likelihood")=1)
        .def("set_label_proposer", &VertexLabeledGraphReconstructionMCMC<Label>::setLabelProposer, py::arg("label_proposer"))
        .def("get_label_proposer", &VertexLabeledGraphReconstructionMCMC<Label>::getLabelProposer)
        .def("get_vertex_labels", &VertexLabeledGraphReconstructionMCMC<Label>::getVertexLabels)
        .def("get_log_acceptance_prob_from_label_move", &VertexLabeledGraphReconstructionMCMC<Label>::getLogAcceptanceProbFromLabelMove, py::arg("move"))
        .def("apply_label_move", &VertexLabeledGraphReconstructionMCMC<Label>::applyLabelMove, py::arg("move"))
        ;
}

}

#endif
