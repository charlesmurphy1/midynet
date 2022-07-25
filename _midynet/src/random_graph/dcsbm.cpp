#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;


void DegreeCorrectedStochasticBlockModelFamily::sampleState(){
    MultiGraph graph = generateDCSBM(getLabels(), getLabelGraph(), getDegrees());
    setGraph( graph );
    computationFinished();
}

void DegreeCorrectedStochasticBlockModelFamily::checkSelfConsistency() const{
    m_degreePriorPtr->checkSelfConsistency();
    checkGraphConsistencyWithLabelGraph("DegreeCorrectedStochasticBlockModelFamily", m_graph, getLabels(), getLabelGraph());
    checkGraphConsistencyWithDegreeSequence("DegreeCorrectedStochasticBlockModelFamily", m_graph, getDegrees());
}
