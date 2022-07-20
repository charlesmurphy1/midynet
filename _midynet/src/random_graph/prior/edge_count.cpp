#include "FastMIDyNet/random_graph/prior/edge_count.h"
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

const double EdgeCountPoissonPrior::getLogLikelihoodFromState(const size_t& state) const {
    return logPoissonPMF(state, m_mean);
}

void EdgeCountPoissonPrior::checkSelfConsistency() const {
    if (m_mean < 0)
        throw ConsistencyError("EdgeCountPoissonPrior: Negative mean `" + std::to_string(m_mean) + "`.");

    if (m_state < 0)
        throw ConsistencyError("EdgeCountPoissonPrior: Negative state `" + std::to_string(m_state) + "`.");
}

// void EdgeCountMultisetPrior::sampleState(){
//     size_t E = m_maxWeightEdgeCount;
//     std::uniform_int_distribution<int> flipDist(0, 1);
//     std::uniform_real_distribution<double> uniformDist(0, 1);
//
//     for(size_t i = 0; i < m_iteration; ++i){
//         int dE = 2 * flipDist(rng) - 1;
//
//         if (static_cast<int>(E + dE) > 0 and E + dE <= m_maxEdgeCount){
//             double dS = this->getWeight(E + dE) - this->getWeight(E);
//
//             if ( uniformDist(rng) < exp(dS) ) E += dE;
//         }
//     }
//     setState(E);
// }
//
// double EdgeCountMultisetPrior::getLogNormalization() const {
//     double Z = 0, maxLogZ = getWeight(m_maxWeightEdgeCount);
//     for (size_t i = 0; i <= m_maxEdgeCount; ++i){
//         Z += exp(getWeight(i) - maxLogZ);
//     }
//     return log(Z) + maxLogZ;
// }

}
