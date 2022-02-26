#ifndef FAST_MIDYNET_PYWRAPPER_INIT_COLLECTORS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_COLLECTORS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/mcmc/callbacks/collector.h"
#include "FastMIDyNet/mcmc/python/callback.hpp"
#include "FastMIDyNet/utility/functions.h"
// #include "FastMIDyNet/utility/distance.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initCollectors(py::module& m){
    /* Collect base classes */
    py::class_<Collector, CallBack, PyCollector<>>(m, "Collector")
        .def(py::init<>())
        .def("collect", &Collector::collect)
        .def("clear", &Collector::clear);

    py::class_<SweepCollector, Collector, PyCollector<SweepCollector>>(m, "SweepCollector")
        .def(py::init<>());

    py::class_<StepCollector, Collector, PyCollector<StepCollector>>(m, "StepCollector")
        .def(py::init<>());

    /* Graph collector classes */
    py::class_<CollectGraphOnSweep, SweepCollector>(m, "CollectGraphOnSweep")
        .def(py::init<>())
        .def("get_graphs", &CollectGraphOnSweep::getGraphs);

    py::class_<CollectEdgeMultiplicityOnSweep, SweepCollector>(m, "CollectEdgeMultiplicityOnSweep")
        .def(py::init<>())
        .def("get_marginal_entropy", &CollectEdgeMultiplicityOnSweep::getMarginalEntropy)
        .def("get_total_count", &CollectEdgeMultiplicityOnSweep::getTotalCount)
        .def("get_edge_observation_count", [](const CollectEdgeMultiplicityOnSweep& self, size_t v, size_t u){
                return self.getEdgeObservationCount(getOrderedPair<BaseGraph::VertexIndex>({u, v}));
            }, py::arg("v"), py::arg("u"))
        .def("get_edge_count_prob", &CollectEdgeMultiplicityOnSweep::getEdgeObservationCount)
        .def("get_edge_count_prob", [](const CollectEdgeMultiplicityOnSweep& self, size_t v, size_t u, size_t count){
                return self.getEdgeCountProb(getOrderedPair<BaseGraph::VertexIndex>({u, v}), count);
            }, py::arg("v"), py::arg("u"), py::arg("count"))
        .def("get_edge_probs", &CollectEdgeMultiplicityOnSweep::getEdgeProbs)
        .def("get_log_posterior_estimate", py::overload_cast<const MultiGraph&>(&CollectEdgeMultiplicityOnSweep::getLogPosteriorEstimate), py::arg("graph"))
        .def("get_log_posterior_estimate", py::overload_cast<>(&CollectEdgeMultiplicityOnSweep::getLogPosteriorEstimate))
        ;
    py::class_<CollectPartitionOnSweep, SweepCollector>(m, "CollectPartitionOnSweep")
        .def(py::init<>())
        .def("get_partitions", &CollectPartitionOnSweep::getPartitions);

    py::class_<WriteGraphToFileOnSweep, SweepCollector>(m, "WriteGraphToFileOnSweep")
        .def(py::init<std::string, std::string>(), py::arg("filename"), py::arg("ext")=".b");

    /* Metrics collector classes */
    py::class_<CollectLikelihoodOnSweep, SweepCollector>(m, "CollectLikelihoodOnSweep")
        .def(py::init<>())
        .def("get_log_likelihoods", &CollectLikelihoodOnSweep::getLogLikelihoods);

    py::class_<CollectPriorOnSweep, SweepCollector>(m, "CollectPriorOnSweep")
        .def(py::init<>())
        .def("get_log_priors", &CollectPriorOnSweep::getLogPriors);

    py::class_<CollectJointOnSweep, SweepCollector>(m, "CollectJointOnSweep")
        .def(py::init<>())
        .def("get_log_joints", &CollectJointOnSweep::getLogJoints);

    // py::class_<CollectGraphDistance, Collector>(m, "CollectGraphDistance")
    //     .def(py::init<const GraphDistance&>(), py::arg("distance"))
    //     .def("get_distances", &CollectGraphDistance::getCollectedDistances);
}

}

#endif
