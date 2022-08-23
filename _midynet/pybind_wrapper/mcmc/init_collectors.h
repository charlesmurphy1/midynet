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
    declareCollectorBaseClass<PartitionReconstructionMCMC>(m, "BlockCollector");
    declareCollectorBaseClass<NestedPartitionReconstructionMCMC>(m, "NestedBlockCollector");
    declareCollectorBaseClass<GraphReconstructionMCMCBase>(m, "GraphReconstructionCollector");
    declareCollectorBaseClass<BlockLabeledGraphReconstructionMCMCBase>(m, "BlockLabeledGraphReconstructionCollector");
    declareCollectorBaseClass<NestedBlockLabeledGraphReconstructionMCMCBase>(m, "NestedBlockLabeledGraphReconstructionCollector");

    /* StepCollect base classes */
    declareCollectorSubClass<StepCollector<MCMC>, Collector<MCMC>, PyCollector<MCMC, StepCollector<MCMC>>>(m, "StepCollector");
    declareCollectorSubClass<StepCollector<PartitionReconstructionMCMC>, Collector<PartitionReconstructionMCMC>, PyCollector<PartitionReconstructionMCMC, StepCollector<PartitionReconstructionMCMC>>>(m, "PartitionReconstructionStepCollector");
    declareCollectorSubClass<StepCollector<NestedPartitionReconstructionMCMC>, Collector<NestedPartitionReconstructionMCMC>, PyCollector<NestedPartitionReconstructionMCMC, StepCollector<NestedPartitionReconstructionMCMC>>>(m, "NestedPartitionReconstructionStepCollector");
    declareCollectorSubClass<StepCollector<GraphReconstructionMCMCBase>, Collector<GraphReconstructionMCMCBase>, PyCollector<GraphReconstructionMCMCBase, StepCollector<GraphReconstructionMCMCBase>>>(m, "GraphReconstructionStepCollector");
    declareCollectorSubClass<StepCollector<BlockLabeledGraphReconstructionMCMCBase>, Collector<BlockLabeledGraphReconstructionMCMCBase>,  PyCollector<BlockLabeledGraphReconstructionMCMCBase, StepCollector<BlockLabeledGraphReconstructionMCMCBase>>>(m, "BlockLabeledGraphReconstructionStepCollector");
    declareCollectorSubClass<StepCollector<NestedBlockLabeledGraphReconstructionMCMCBase>, Collector<NestedBlockLabeledGraphReconstructionMCMCBase>,  PyCollector<NestedBlockLabeledGraphReconstructionMCMCBase, StepCollector<NestedBlockLabeledGraphReconstructionMCMCBase>>>(m, "NestedBlockLabeledGraphReconstructionStepCollector");

    /* SweepCollector base classes */
    declareCollectorSubClass<SweepCollector<MCMC>, Collector<MCMC>, PyCollector<MCMC, SweepCollector<MCMC>>>(m, "SweepCollector");
    declareCollectorSubClass<SweepCollector<PartitionReconstructionMCMC>, Collector<PartitionReconstructionMCMC>, PyCollector<PartitionReconstructionMCMC, SweepCollector<PartitionReconstructionMCMC>>>(m, "PartitionReconstructionSweepCollector");
    declareCollectorSubClass<SweepCollector<NestedPartitionReconstructionMCMC>, Collector<NestedPartitionReconstructionMCMC>, PyCollector<NestedPartitionReconstructionMCMC, SweepCollector<NestedPartitionReconstructionMCMC>>>(m, "NestedPartitionReconstructionSweepCollector");
    declareCollectorSubClass<SweepCollector<GraphReconstructionMCMCBase>, Collector<GraphReconstructionMCMCBase>, PyCollector<GraphReconstructionMCMCBase, SweepCollector<GraphReconstructionMCMCBase>>>(m, "GraphReconstructionSweepCollector");
    declareCollectorSubClass<SweepCollector<BlockLabeledGraphReconstructionMCMCBase>, Collector<BlockLabeledGraphReconstructionMCMCBase>,  PyCollector<BlockLabeledGraphReconstructionMCMCBase, SweepCollector<BlockLabeledGraphReconstructionMCMCBase>>>(m, "BlockLabeledGraphReconstructionSweepCollector");
    declareCollectorSubClass<SweepCollector<NestedBlockLabeledGraphReconstructionMCMCBase>, Collector<NestedBlockLabeledGraphReconstructionMCMCBase>,  PyCollector<NestedBlockLabeledGraphReconstructionMCMCBase, SweepCollector<NestedBlockLabeledGraphReconstructionMCMCBase>>>(m, "NestedBlockLabeledGraphReconstructionSweepCollector");

    /* Graph collector classes */
    declareCollectorSubClass<CollectGraphOnSweep<GraphReconstructionMCMCBase>, SweepCollector<GraphReconstructionMCMCBase>>(m, "_CollectGraphOnSweep")
        .def("get_data", &CollectGraphOnSweep<GraphReconstructionMCMCBase>::getData);
    declareCollectorSubClass<CollectBlockLabeledGraphOnSweep, BlockLabeledGraphReconstructionSweepCollector>(m, "_CollectBlockLabeledGraphOnSweep")
        .def("get_data", &CollectBlockLabeledGraphOnSweep::getData);
    declareCollectorSubClass<CollectNestedBlockLabeledGraphOnSweep, NestedBlockLabeledGraphReconstructionSweepCollector>(m, "_CollectNestedBlockLabeledGraphOnSweep")
        .def("get_data", &CollectNestedBlockLabeledGraphOnSweep::getData);

    /* Edge multiplicity collector classes */
    declareEdgeMultiplicityCollector<GraphReconstructionMCMCBase>(m, "_CollectEdgeMultiplicityOnSweep");
    declareEdgeMultiplicityCollector<BlockLabeledGraphReconstructionMCMCBase>(m, "_CollectBlockLabeledEdgeMultiplicityOnSweep");
    declareEdgeMultiplicityCollector<NestedBlockLabeledGraphReconstructionMCMCBase>(m, "_CollectNestedBlockLabeledEdgeMultiplicityOnSweep");

    /* Partition collector classes */
    declareCollectorSubClass<CollectPartitionOnSweepForCommunity, PartitionReconstructionSweepCollector>(m, "_CollectPartitionOnSweepForCommunity")
        .def("get_data", &CollectPartitionOnSweepForCommunity::getData);
    declareCollectorSubClass<CollectPartitionOnSweepForReconstruction, BlockLabeledGraphReconstructionSweepCollector>(m, "_CollectPartitionOnSweepForReconstruction")
        .def("get_data", &CollectPartitionOnSweepForReconstruction::getData);
    declareCollectorSubClass<CollectNestedPartitionOnSweepForCommunity, NestedPartitionReconstructionSweepCollector>(m, "_CollectNestedPartitionOnSweepForCommunity")
        .def("get_data", &CollectNestedPartitionOnSweepForCommunity::getData);
    declareCollectorSubClass<CollectNestedPartitionOnSweepForReconstruction, NestedBlockLabeledGraphReconstructionSweepCollector>(m, "_CollectNestedPartitionOnSweepForReconstruction")
        .def("get_data", &CollectNestedPartitionOnSweepForReconstruction::getData);


    /* MCMC metrics collector classes */
    declareCollectorSubClass<CollectLikelihoodOnSweep, SweepCollector<MCMC>>(m, "CollectLikelihoodOnSweep")
        .def("get_data", &CollectLikelihoodOnSweep::getData);

    declareCollectorSubClass<CollectPriorOnSweep, SweepCollector<MCMC>>(m, "CollectPriorOnSweep")
        .def("get_data", &CollectPriorOnSweep::getData);

    declareCollectorSubClass<CollectJointOnSweep, SweepCollector<MCMC>>(m, "CollectJointOnSweep")
        .def("get_data", &CollectJointOnSweep::getData);

    // py::class_<WriteGraphToFileOnSweep, SweepCollector>(m, "WriteGraphToFileOnSweep")
    //     .def(py::init<std::string, std::string>(), py::arg("filename"), py::arg("ext")=".b");
    // py::class_<CollectGraphDistance, Collector>(m, "CollectGraphDistance")
    //     .def(py::init<const GraphDistance&>(), py::arg("distance"))
    //     .def("get_distances", &CollectGraphDistance::getCollectedDistances);
}

}

#endif
