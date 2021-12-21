#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_blockcount.h"
#include "init_vertexcount.h"
#include "init_block.h"
#include "init_edgecount.h"
#include "init_edgematrix.h"
#include "init_degree.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initSBMPrior(py::module& m){
    initBlockCountPrior(m);
    initVertexCountPrior(m);
    initBlockPrior(m);
    initEdgeCountPrior(m);
    initEdgeMatrixPrior(m);
    initDegreePrior(m);
}

}

#endif
