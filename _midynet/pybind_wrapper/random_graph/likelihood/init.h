#ifndef FAST_MIDYNET_PYWRAPPER_LIKELIHOOD_INIT_H
#define FAST_MIDYNET_PYWRAPPER_LIKELIHOOD_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_likelihood.h"
#include "FastMIDyNet/random_graph/likelihood/erdosrenyi.h"
#include "FastMIDyNet/random_graph/likelihood/configuration.h"
#include "FastMIDyNet/random_graph/likelihood/sbm.h"
#include "FastMIDyNet/random_graph/likelihood/dcsbm.h"

namespace py = pybind11;
namespace FastMIDyNet{


void initLikelihoods(py::module& m){
    initGraphLikelihoods(m);
    py::class_<ErdosRenyiLikelihood, GraphLikelihoodModel>(m, "ErdosRenyiLikelihood");
    py::class_<ConfigurationModelLikelihood, GraphLikelihoodModel>(m, "ConfigurationModelLikelihood");
    py::class_<StochasticBlockModelLikelihood, VertexLabeledGraphLikelihoodModel<BlockIndex>>(m, "StochasticBlockModelLikelihood");
    py::class_<StubLabeledStochasticBlockModelLikelihood, StochasticBlockModelLikelihood>(m, "StubLabeledStochasticBlockModelLikelihood");
    py::class_<UniformStochasticBlockModelLikelihood, StochasticBlockModelLikelihood>(m, "UniformStochasticBlockModelLikelihood");
    py::class_<DegreeCorrectedStochasticBlockModelLikelihood, VertexLabeledGraphLikelihoodModel<BlockIndex>>(m, "DegreeCorrectedStochasticBlockModelLikelihood");
}

}

#endif
