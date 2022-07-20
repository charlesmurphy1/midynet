#include <algorithm>
#include <string>

#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


namespace FastMIDyNet{


// size_t BlockCountPrior::getStateAfterLabelMove(const BlockMove& move) const {
//     return getState() + move.addedLabels;
// };

void BlockCountPoissonPrior::sampleState() {
    auto blockCount = 0;
    while (blockCount == 0) // zero-truncated Poisson sampling
        blockCount = m_poissonDistribution(rng);
    setState(blockCount);
};

const double BlockCountPoissonPrior::getLogLikelihoodFromState(const size_t& state) const {
    return logZeroTruncatedPoissonPMF(state, m_mean);
};

void BlockCountPoissonPrior::checkSelfConsistency() const {
    if (m_mean < 0)
        throw ConsistencyError("BlockCountPoissonPrior: Negative mean `" + std::to_string(m_mean) + "`.");

    if (m_state<=0)
        throw ConsistencyError("BlockCountPoissonPrior: Non-positive state `" + std::to_string(m_state) + "`.");
};

void BlockCountUniformPrior::checkMin() const {
    if (m_min < 0)
        throw ConsistencyError("BlockCountPoissonPrior: Negative mean `" + std::to_string(m_min) + "`.");
}

void BlockCountUniformPrior::checkMax() const {
    if (m_max < m_min)
        throw ConsistencyError("BlockCountUniformPrior: `max` must be greater than or equal to `min` :"
            + std::to_string(m_min) + ">" + std::to_string(m_max) + ".");
}
void BlockCountUniformPrior::checkSelfConsistency() const {
    checkMin();
    checkMax();
    if (m_state < m_min || m_state > m_max)
        throw ConsistencyError("BlockCountUniformPrior: Inconsistent state " + std::to_string(m_state)
            + ", must be within [" + std::to_string(m_min) + ", " + std::to_string(m_max) + "]." );


};

}
