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

void BlockCountPoissonPrior::checkSelfSafety() const {
    if (m_mean < 0)
        throw SafetyError("BlockCountPoissonPrior", "m_mean", "<0");

    if (m_state<=0)
        throw SafetyError("BlockCountPoissonPrior", "m_state", "<1");
};

void BlockCountUniformPrior::checkMin() const {
    if (m_min < 0)
        throw SafetyError("BlockCountUniformPrior", "m_min", "<0");
}

void BlockCountUniformPrior::checkMax() const {
    if (m_max < m_min)
        throw SafetyError("BlockCountUniformPrior: `max` must be greater than or equal to `min` :"
            + std::to_string(m_min) + ">" + std::to_string(m_max) + ".");
}
void BlockCountUniformPrior::checkSelfSafety() const {
    checkMin();
    checkMax();
    // if (m_state < m_min || m_state > m_max)
    //     throw SaftetyError(
    //         "BlockCountUniformPrior"
    //
    //         : Inconsistent state " + std::to_string(m_state)
    //         + ", must be within [" + std::to_string(m_min) + ", " + std::to_string(m_max) + "]."
    //     );
};

}
