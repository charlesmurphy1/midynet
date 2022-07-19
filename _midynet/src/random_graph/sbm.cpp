#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;

void StochasticBlockModelFamily::sampleState(){ setGraph(generateStubLabeledSBM(getLabels(), getLabelGraph().getAdjacencyMatrix())); }


void StochasticBlockModelFamily::checkSelfConsistency() const{
    m_edgeMatrixPriorPtr->checkSelfConsistency();
    checkGraphConsistencyWithEdgeMatrix("StochasticBlockModelFamily", m_graph, getLabels(), getLabelGraph());
}
