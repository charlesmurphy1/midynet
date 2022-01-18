#ifndef FAST_MIDYNET_PYWRAPPER_INIT_MCMCBASECLASS_H
#define AST_MIDYNET_PYWRAPPER_INIT_MCMCBASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/mcmc/python/mcmc.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initMCMCBaseClass(py::module& m){
    py::class_<MCMC, PyMCMC<>>(m, "MCMC")
        .def(py::init<>())
        .def(py::init<std::vector<CallBack*>>(), py::arg("callbacks"))
        .def(py::init<const CallBackList&>(), py::arg("callback_list"))
        .def("get_last_log_joint_ratio", &MCMC::getLastLogJointRatio)
        .def("get_last_log_acceptance", &MCMC::getLastLogAcceptance)
        .def("is_last_accepted", &MCMC::isLastAccepted)
        .def("get_graph", &MCMC::getGraph)
        .def("get_blocks", &MCMC::getBlocks)
        .def("has_state", &MCMC::hasState)
        .def("get_num_steps", &MCMC::getNumSteps)
        .def("get_num_sweeps", &MCMC::getNumSweeps)
        .def("get_log_likelihood", &MCMC::getLogLikelihood)
        .def("get_log_prior", &MCMC::getLogPrior)
        .def("get_log_joint", &MCMC::getLogJoint)
        .def("sample", &MCMC::sample)
        .def("add_callback", &MCMC::addCallBack, py::arg("callback"))
        .def("remove_callback", &MCMC::removeCallBack, py::arg("idx"))
        .def("pop_callback", &MCMC::popCallBack)
        .def("set_up", &MCMC::setUp)
        .def("tear_down", &MCMC::tearDown)
        .def("do_metropolis_hastings_step", &MCMC::doMetropolisHastingsStep)
        .def("do_MH_sweep", &MCMC::doMHSweep, py::arg("burn")=1)
        ;

    py::class_<RandomGraphMCMC, MCMC>(m, "RandomGraphMCMC")
        .def(py::init<RandomGraph&, EdgeProposer&, BlockProposer&, double, double, const CallBackList&>(),
            py::arg("random_graph"), py::arg("edge_proposer"), py::arg("block_proposer"),
            py::arg("beta_likelihood")=1, py::arg("beta_prior")=1, py::arg("callbacks"))
        .def(py::init<EdgeProposer&, BlockProposer&, double, double, const CallBackList&>(),
            py::arg("edge_proposer"), py::arg("block_proposer"),
            py::arg("beta_likelihood")=1, py::arg("beta_prior")=1,
            py::arg("callbacks"))
        .def(py::init<EdgeProposer&, BlockProposer&, double, double>(),
            py::arg("edge_proposer"), py::arg("block_proposer"),
            py::arg("beta_likelihood")=1, py::arg("beta_prior")=1 )
        .def("get_beta_prior", &RandomGraphMCMC::getBetaPrior)
        .def("set_beta_prior", &RandomGraphMCMC::setBetaPrior, py::arg("beta_prior"))
        .def("get_beta_likelihood", &RandomGraphMCMC::getBetaLikelihood)
        .def("set_beta_likelihood", &RandomGraphMCMC::setBetaLikelihood, py::arg("beta_likelihood"))
        .def("sample", &RandomGraphMCMC::sample )
        .def("sample_graph_only", &RandomGraphMCMC::sampleGraphOnly )
        .def("get_random_graph", &RandomGraphMCMC::getRandomGraph )
        .def("set_random_graph", &RandomGraphMCMC::setRandomGraph, py::arg("random_graph") )
        .def("get_edge_proposer", &RandomGraphMCMC::getEdgeProposer )
        .def("get_block_proposer", &RandomGraphMCMC::getBlockProposer )
        .def("propose_edge_move", &RandomGraphMCMC::proposeEdgeMove )
        .def("get_log_proposal_prob_ratio_from_graph_move", &RandomGraphMCMC::getLogProposalProbRatioFromGraphMove, py::arg("move") )
        .def("get_log_proposal_prob_ratio_from_block_move", &RandomGraphMCMC::getLogProposalProbRatioFromBlockMove, py::arg("move") )
        .def("update_probabilities_from_graph_move", &RandomGraphMCMC::updateProbabilitiesFromGraphMove, py::arg("move") )
        .def("update_probabilities_from_block_move", &RandomGraphMCMC::updateProbabilitiesFromBlockMove, py::arg("move") )
        ;

    py::class_<DynamicsMCMC, MCMC>(m, "DynamicsMCMC")
        .def(py::init<Dynamics&, RandomGraphMCMC&, double, double, double, const CallBackList&>(),
            py::arg("dynamics"), py::arg("random_graph_mcmc"),
            py::arg("beta_likelihood")=1, py::arg("beta_prior")=1,
            py::arg("sample_graph_prior")=0.5, py::arg("callbacks"))
        .def(py::init<Dynamics&, RandomGraphMCMC&, double, double, double>(),
            py::arg("dynamics"), py::arg("random_graph_mcmc"),
            py::arg("beta_likelihood")=1, py::arg("beta_prior")=1,
            py::arg("sample_graph_prior")=0.5)
        .def("get_beta_prior", &DynamicsMCMC::getBetaPrior)
        .def("set_beta_prior", &DynamicsMCMC::setBetaPrior, py::arg("beta_prior"))
        .def("get_beta_likelihood", &DynamicsMCMC::getBetaLikelihood)
        .def("set_beta_likelihood", &DynamicsMCMC::setBetaLikelihood, py::arg("beta_likelihood"))
        .def("sample", &DynamicsMCMC::sample )
        .def("sample_state", &DynamicsMCMC::sampleState )
        .def("sample_graph", &DynamicsMCMC::sampleGraph )
        .def("sample_graph_only", &DynamicsMCMC::sampleGraphOnly )
        .def("get_dynamics", &DynamicsMCMC::getDynamics )
        .def("get_random_graph_mcmc", &DynamicsMCMC::getRandomGraphMCMC )
        .def("get_graph", &DynamicsMCMC::getGraph )
        .def("set_graph", &DynamicsMCMC::setGraph )
        ;
}

}

#endif
