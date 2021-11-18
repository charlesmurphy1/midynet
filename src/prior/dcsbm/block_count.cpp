#include <algorithm>

#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet{


size_t BlockCountPrior::getStateAfterMove(const BlockMove& move) const {
    size_t newState = getState();
    if (newState <= move.nextBlockIdx)
        newState = move.nextBlockIdx + 1;
    return newState;
};

size_t BlockCountPoissonPrior::sample() {
    return m_poissonDistribution(rng);
};

double BlockCountPoissonPrior::getLogLikelihood(const size_t& state) const {
    return logPoissonPMF(state, m_mean);
};

void BlockCountPoissonPrior::checkSelfConsistency() const {
    if (m_mean<=0)
        throw ConsistencyError("BlockCountPoissonPrior: Negative mean.");
};

}
