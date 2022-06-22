#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_movetypes.h"
#include "init_proposer.h"
#include "init_sampler.h"
#include "init_edge.h"
#include "init_label.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initProposer(py::module& m){
    initMoveTypes(m);
    initProposerBaseClass(m);

    auto sampler = m.def_submodule("sampler");
    initSampler(sampler);

    auto edge_proposer = m.def_submodule("edge");
    initEdgeProposer(edge_proposer);

    auto label_proposer = m.def_submodule("label");
    initLabelProposer(label_proposer);
}

}

#endif
