#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utility/init_utility.h"
#include "init_exceptions.h"
#include "init_generator.h"
#include "init_rng.h"
#include "prior/init_prior.h"
#include "random_graph/init.h"
#include "dynamics/init.h"
#include "proposer/init.h"
#include "mcmc/init.h"

namespace py = pybind11;
namespace FastMIDyNet{

PYBIND11_MODULE(_midynet, m) {
    py::module_::import("basegraph");

    py::module utility = m.def_submodule("utility");
    initUtility( utility );
    initGenerators( utility );
    initRNG( utility );
    initExceptions( utility );

    py::module prior = m.def_submodule("prior");
    initPrior( prior );

    py::module random_graph = m.def_submodule("random_graph");
    initRandomGraph( random_graph );

    py::module dynamics = m.def_submodule("dynamics");
    initDynamics( dynamics );

    py::module proposer = m.def_submodule("proposer");
    initProposer( proposer );

    py::module mcmc = m.def_submodule("mcmc");
    initMCMC( mcmc );
}

}
