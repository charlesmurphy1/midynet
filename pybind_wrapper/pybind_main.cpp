#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dynamics/init_dynamics.h"
#include "mcmc/init_mcmc.h"
#include "prior/init_prior.h"
#include "proposer/init_proposer.h"
#include "random_graph/init_randomgraph.h"
#include "utility/init_utility.h"
#include "init_exceptions.h"
#include "init_generator.h"
#include "init_rng.h"

namespace py = pybind11;


PYBIND11_MODULE(fast_midynet, m) {
    py::module dynamics = m.def_submodule("_dynamics");
    initDynamics( dynamics );

    py::module mcmc = m.def_submodule("_mcmc");
    initMCMC( mcmc );

    py::module prior = m.def_submodule("_prior");
    initPrior( prior );

    py::module proposer = m.def_submodule("_proposer");
    initProposer( proposer );

    py::module random_graph = m.def_submodule("_random_graph");
    initRandomGraph( random_graph );

    py::module utility = m.def_submodule("_utility");
    initUtility( utility );
    initGenerators( utility );
    initRNG( utility );
    initExceptions( utility );
}
