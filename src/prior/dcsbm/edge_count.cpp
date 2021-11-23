#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet{

size_t EdgeCountPrior::getStateAfterGraphMove(const GraphMove& move) const {
    long edgeNumberDifference = (long) move.addedEdges.size() - (long) move.removedEdges.size();
    if ((long) m_state + edgeNumberDifference < 0)
        throw ConsistencyError("EdgeCountPoissonPrior: Removing more edges than present in graph.");
    return m_state + edgeNumberDifference;
}

void EdgeCountPoissonPrior::sampleState() {
    setState( m_poissonDistribution(rng) );
}

double EdgeCountPoissonPrior::getLogLikelihoodFromState(const size_t& state) const {
    return logPoissonPMF(state, m_mean);
}

void EdgeCountPoissonPrior::checkSelfConsistency() const {
    if (m_mean < 0)
        throw ConsistencyError("EdgeCountPoissonPrior: Negative mean `" + std::to_string(m_mean) + "`.");

    if (m_state < 0)
        throw ConsistencyError("EdgeCountPoissonPrior: Negative state `" + std::to_string(m_state) + "`.");
}

}
