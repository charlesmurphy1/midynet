#ifndef FAST_MIDYNET_PYWRAPPER_INIT_COLLECTORS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_COLLECTORS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/mcmc/callbacks/collector.hpp"
#include "FastMIDyNet/mcmc/python/callback.hpp"
#include "FastMIDyNet/utility/functions.h"
// #include "FastMIDyNet/utility/distance.h"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename MCMCType>
py::class_<Collector<MCMCType>, CallBack<MCMCType>, PyCollector<MCMCType>> declareCollectorBaseClass(py::module& m, std::string pyName){
    return py::class_<Collector<MCMCType>, CallBack<MCMCType>, PyCollector<MCMCType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("collect", &Collector<MCMCType>::collect)
        ;
}

template<typename SubClass, typename ...BaseClasses>
py::class_<SubClass, BaseClasses...> declareCollectorSubClass(py::module& m, std::string pyName){
    return py::class_<SubClass, BaseClasses...>(m, pyName.c_str())
        .def(py::init<>())
        ;
}

template<typename MCMCType>
py::class_<CollectEdgeMultiplicityOnSweep<MCMCType>, SweepCollector<MCMCType>> declareEdgeMultiplicityCollector(py::module& m, std::string pyName){
    return py::class_<CollectEdgeMultiplicityOnSweep<MCMCType>, SweepCollector<MCMCType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("get_marginal_entropy", &CollectEdgeMultiplicityOnSweep<MCMCType>::getMarginalEntropy)
        .def("get_total_count", &CollectEdgeMultiplicityOnSweep<MCMCType>::getTotalCount)
        .def("get_edge_observation_count", [](const CollectEdgeMultiplicityOnSweep<MCMCType>& self, size_t v, size_t u){
                return self.getEdgeObservationCount(getOrderedPair<BaseGraph::VertexIndex>({u, v}));
            }, py::arg("v"), py::arg("u"))
        .def("get_edge_count_prob", &CollectEdgeMultiplicityOnSweep<MCMCType>::getEdgeObservationCount)
        .def("get_edge_count_prob", [](const CollectEdgeMultiplicityOnSweep<MCMCType>& self, size_t v, size_t u, size_t count){
                return self.getEdgeCountProb(getOrderedPair<BaseGraph::VertexIndex>({u, v}), count);
            }, py::arg("v"), py::arg("u"), py::arg("count"))
        .def("get_edge_probs", &CollectEdgeMultiplicityOnSweep<MCMCType>::getEdgeProbs)
        .def("get_log_posterior_estimate", py::overload_cast<const MultiGraph&>(&CollectEdgeMultiplicityOnSweep<MCMCType>::getLogPosteriorEstimate), py::arg("graph"))
        .def("get_log_posterior_estimate", py::overload_cast<>(&CollectEdgeMultiplicityOnSweep<MCMCType>::getLogPosteriorEstimate))
        ;

}

void initCollectors(py::module& m){
    /* Collect base classes */
    declareCollectorBaseClass<MCMC>(m, "Collector");
    declareCollectorBaseClass<BlockLabelMCMC>(m, "BlockCollector");
    declareCollectorBaseClass<GraphReconstructionMCMC<>>(m, "GraphReconstructionCollector");
    declareCollectorBaseClass<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>(m, "BlockLabeledGraphReconstructionCollector");

    /* StepCollect base classes */
    declareCollectorSubClass<StepCollector<MCMC>, Collector<MCMC>, PyCollector<MCMC, StepCollector<MCMC>>>(m, "StepCollector");
    declareCollectorSubClass<StepCollector<BlockLabelMCMC>, Collector<BlockLabelMCMC>, PyCollector<BlockLabelMCMC, StepCollector<BlockLabelMCMC>>>(m, "BlockStepCollector");
    declareCollectorSubClass<StepCollector<GraphReconstructionMCMC<>>, Collector<GraphReconstructionMCMC<>>, PyCollector<GraphReconstructionMCMC<>, StepCollector<GraphReconstructionMCMC<>>>>(m, "GraphReconstructionStepCollector");
    declareCollectorSubClass<StepCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>, Collector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>,  PyCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>, StepCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>>>(m, "BlockLabeledGraphReconstructionStepCollector");

    /* SweepCollector base classes */
    declareCollectorSubClass<SweepCollector<MCMC>, Collector<MCMC>, PyCollector<MCMC, SweepCollector<MCMC>>>(m, "SweepCollector");
    declareCollectorSubClass<SweepCollector<BlockLabelMCMC>, Collector<BlockLabelMCMC>, PyCollector<BlockLabelMCMC, SweepCollector<BlockLabelMCMC>>>(m, "BlockSweepCollector");
    declareCollectorSubClass<SweepCollector<GraphReconstructionMCMC<>>, Collector<GraphReconstructionMCMC<>>, PyCollector<GraphReconstructionMCMC<>, SweepCollector<GraphReconstructionMCMC<>>>>(m, "GraphReconstructionSweepCollector");
    declareCollectorSubClass<SweepCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>, Collector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>,  PyCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>, SweepCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>>>(m, "BlockLabeledGraphReconstructionSweepCollector");

    /* Graph collector classes */
    declareCollectorSubClass<CollectGraphOnSweep<GraphReconstructionMCMC<>>, SweepCollector<GraphReconstructionMCMC<>>>(m, "CollectGraphOnSweep")
        .def("get_graphs", &CollectGraphOnSweep<GraphReconstructionMCMC<>>::getGraphs);
    declareCollectorSubClass<CollectBlockLabeledGraphOnSweep, BlockLabeledGraphReconstructionSweepCollector>(m, "CollectBlockLabeledGraphOnSweep")
        .def("get_graphs", &CollectBlockLabeledGraphOnSweep::getGraphs);

    /* Edge multiplicity collector classes */
    declareEdgeMultiplicityCollector<GraphReconstructionMCMC<>>(m, "CollectEdgeMultiplicityOnSweep");
    declareEdgeMultiplicityCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>(m, "CollectBlockLabeledEdgeMultiplicityOnSweep");

    /* Partition collector classes */
    declareCollectorSubClass<CollectPartitionOnSweepForCommunity, BlockSweepCollector>(m, "CollectPartitionOnSweepForCommunity")
        .def("get_partitions", &CollectPartitionOnSweepForCommunity::getPartitions);
    declareCollectorSubClass<CollectPartitionOnSweepForReconstruction, BlockLabeledGraphReconstructionSweepCollector>(m, "CollectPartitionOnSweepForReconstruction")
        .def("get_partitions", &CollectPartitionOnSweepForReconstruction::getPartitions);


    /* MCMC metrics collector classes */
    declareCollectorSubClass<CollectLikelihoodOnSweep, SweepCollector<MCMC>>(m, "CollectLikelihoodOnSweep")
        .def("get_log_likelihoods", &CollectLikelihoodOnSweep::getLogLikelihoods);

    declareCollectorSubClass<CollectPriorOnSweep, SweepCollector<MCMC>>(m, "CollectPriorOnSweep")
        .def("get_log_priors", &CollectPriorOnSweep::getLogPriors);

    declareCollectorSubClass<CollectJointOnSweep, SweepCollector<MCMC>>(m, "CollectJointOnSweep")
        .def("get_log_joints", &CollectJointOnSweep::getLogJoints);

    // py::class_<WriteGraphToFileOnSweep, SweepCollector>(m, "WriteGraphToFileOnSweep")
    //     .def(py::init<std::string, std::string>(), py::arg("filename"), py::arg("ext")=".b");
    // py::class_<CollectGraphDistance, Collector>(m, "CollectGraphDistance")
    //     .def(py::init<const GraphDistance&>(), py::arg("distance"))
    //     .def("get_distances", &CollectGraphDistance::getCollectedDistances);
}

}

#endif
