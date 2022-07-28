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

void StochasticBlockModelBase::sampleState(){ setGraph(generateStubLabeledSBM(getLabels(), getLabelGraph())); }


void StochasticBlockModelBase::checkSelfConsistency() const{
    m_labelGraphPriorPtr->checkSelfConsistency();
    checkGraphConsistencyWithLabelGraph("StochasticBlockModelBase", m_graph, getLabels(), getLabelGraph());
}
