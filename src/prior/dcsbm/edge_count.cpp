#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/utility.h"


namespace FastMIDyNet{

size_t EdgeCountPoissonPrior::sample() {
    return m_poissonDistribution(rng);
}

double EdgeCountPoissonPrior::getLogLikelihood(size_t state) const {
    return logPoissonPMF(state, m_mean);
}


void EdgeCountPoissonPrior::checkSelfConsistency() {
    if (m_mean<=0)
        throw "EdgeCountPoissonPrior: Negative mean.";
}

size_t EdgeCountPoissonPrior::getStateAfterMove(const GraphMove& move) const {
    long edgeNumberDifference = (long) move.addedEdges.size() - (long) move.removedEdges.size();
    if ((long) m_state + edgeNumberDifference < 0)
        throw "EdgeCountPoissonPrior: Removing more edges than present in graph.";
    return m_state + edgeNumberDifference;
}

}
