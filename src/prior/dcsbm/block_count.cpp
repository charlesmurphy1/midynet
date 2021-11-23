#include <algorithm>
#include <string>

#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet{


size_t BlockCountPrior::getStateAfterBlockMove(const BlockMove& move) const {
    // size_t newState = getState();
    // if (newState <= move.nextBlockIdx)
    //     newState = move.nextBlockIdx + 1;
    return getState() + move.addedBlocks;
};

void BlockCountPoissonPrior::sampleState() {
    setState(m_poissonDistribution(rng));
};

double BlockCountPoissonPrior::getLogLikelihoodFromState(const size_t& state) const {
    return logPoissonPMF(state, m_mean);
};

void BlockCountPoissonPrior::checkSelfConsistency() const {
    if (m_mean < 0)
        throw ConsistencyError("BlockCountPoissonPrior: Negative mean `" + std::to_string(m_mean) + "`.");

    if (m_state<=0)
        throw ConsistencyError("BlockCountPoissonPrior: Non-positive state `" + std::to_string(m_state) + "`.");
};

}
