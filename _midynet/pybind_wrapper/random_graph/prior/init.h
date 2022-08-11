#ifndef FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_H
#define FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_prior.h"
#include "init_edgecount.h"
#include "init_blockcount.h"
#include "init_block.h"
#include "init_degree.h"
#include "init_labelgraph.h"
#include "init_labeled_degree.h"
#include "init_nestedblocks.h"
#include "init_nestedlabelgraph.h"

namespace py = pybind11;
namespace FastMIDyNet{


void initPriors(py::module& m){
    initPriorBaseClass(m);

    initBlockCountPrior(m);
    initBlockPrior(m);
    initNestedBlockPrior(m);

    initEdgeCountPrior(m);
    initLabelGraphPrior(m);
    initNestedLabelGraphPrior(m);

    initDegreePrior(m);
    initLabeledDegreePrior(m);
}

}

#endif
