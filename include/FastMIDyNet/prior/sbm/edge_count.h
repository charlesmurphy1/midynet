#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:

        using Prior::Prior;
        void samplePriors() {}
        virtual double getLogLikelihoodFromState(const size_t&) const = 0;
        virtual double getLogLikelihood() const { return getLogLikelihoodFromState(m_state); }
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
             return getLogLikelihoodFromState(getStateAfterGraphMove(move)) - getLogLikelihood();
        }
        double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
             return 0;
        }
        double getLogJointRatioFromGraphMove(const GraphMove& move) {
            double ratio = processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromGraphMove(move); }, 0);
            return ratio;
        }
        double getLogJointRatioFromBlockMove(const BlockMove& move) { return 0; }

        void applyGraphMove(const GraphMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterGraphMove(move)); } );
        }
        void applyBlockMove(const BlockMove& move) { }
        size_t getStateAfterGraphMove(const GraphMove& move) const;
};

class EdgeCountDeltaPrior: public EdgeCountPrior{
    size_t m_edgeCount;
public:
    using EdgeCountPrior::EdgeCountPrior;
    EdgeCountDeltaPrior(const size_t& edgeCount): m_edgeCount(edgeCount){ setState(m_edgeCount); }
    void sampleState() { };
    double getLogLikelihoodFromState(const size_t& state) const { if (state == m_state) return 0.; else return -INFINITY; };
    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) { if (move.addedEdges.size() == move.removedEdges.size()) return 0; else return -INFINITY;}
    void checkSelfConsistency() const { };

};

class EdgeCountPoissonPrior: public EdgeCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        using EdgeCountPrior::EdgeCountPrior;
        EdgeCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }
        void sampleState();
        double getLogLikelihoodFromState(const size_t& state) const;
        void checkSelfConsistency() const;

};

// class EdgeCountMultisetPrior: public EdgeCountPrior{
// protected:
//     size_t m_maxWeightEdgeCount;
//     size_t m_maxEdgeCount;
//     size_t m_iteration;
//     double m_logZ;
//
// public:
//     using EdgeCountPrior::EdgeCountPrior;
//     EdgeCountMultisetPrior(size_t maxEdgeCount, size_t iteration=100):
//         m_maxEdgeCount(maxEdgeCount),
//         m_iteration(iteration),
//         m_logZ(getLogNormalization()),
//         m_maxWeightEdgeCount(maxEdgeCount) { }
//
//     void sampleState() ;
//     double getLogLikelihoodFromState(const size_t& E) const { return this->getWeight(E) - m_logZ; }
//     double getLogNormalization() const ;
//     virtual double getWeight(size_t E) const {
//         return logMultisetCoefficient(m_maxEdgeCount, E);
//     }
//     void checkSelfConsistency() const {};
// };
//
// class EdgeCountBinomialPrior: public EdgeCountMultisetPrior{
// public:
//     using EdgeCountMultisetPrior::EdgeCountMultisetPrior;
//     EdgeCountBinomialPrior(size_t maxEdgeCount, size_t iteration=100):
//         EdgeCountMultisetPrior(maxEdgeCount, iteration){ m_maxWeightEdgeCount = maxEdgeCount / 2; }
//     double getWeight(size_t E) const {
//         return logBinomialCoefficient(m_maxEdgeCount, E);
//     }
// };

}

#endif
