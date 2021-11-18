#include <algorithm>

#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/utility.h"


namespace FastMIDyNet{


size_t BlockCountPrior::getStateAfterMove(const std::vector<BlockMove>& move) const {
    size_t newState = m_state;
    for (auto vertexBlockMove : move){
        if (newState <= vertexBlockMove.nextBlockIdx)
            newState = vertexBlockMove.nextBlockIdx + 1;
    }
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
