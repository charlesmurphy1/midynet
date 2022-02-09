#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_movetypes.h"
#include "init_proposer.h"
#include "init_sampler.h"
#include "init_edgeproposer.h"
#include "init_blockproposer.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initProposer(py::module& m){
    initMoveTypes(m);
    initProposerBaseClass(m);

    auto sampler = m.def_submodule("sampler");
    initSampler(sampler);

    auto edge_proposer = m.def_submodule("edge_proposer");
    initEdgeProposer(edge_proposer);

    auto block_proposer = m.def_submodule("block_proposer");
    initBlockProposer(block_proposer);
}

}

#endif
