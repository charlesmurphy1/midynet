#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
protected:
    void _samplePriors() override {}
    const double _getLogPrior() const override { return 0; }
    void _applyGraphMove(const GraphMove& move) override { setState(getStateAfterGraphMove(move)); }

    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override {
        return 0;
    }

public:
    using Prior<size_t>::Prior;
    virtual const double getLogLikelihoodFromState(const size_t&) const = 0;
    const double getLogLikelihood() const override { return getLogLikelihoodFromState(m_state); }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
         return getLogLikelihoodFromState(getStateAfterGraphMove(move)) - getLogLikelihood();
    }


    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return 0; }


    size_t getStateAfterGraphMove(const GraphMove& move) const;
    void checkSelfSafety()const override {}
};

class EdgeCountDeltaPrior: public EdgeCountPrior{
private:
    size_t m_edgeCount;
public:
    EdgeCountDeltaPrior(){}
    EdgeCountDeltaPrior(const size_t& edgeCount): m_edgeCount(edgeCount){ setState(m_edgeCount); }
    EdgeCountDeltaPrior(const EdgeCountDeltaPrior& other): m_edgeCount(other.m_edgeCount){ setState(m_edgeCount); }
    virtual ~EdgeCountDeltaPrior(){}
    const EdgeCountDeltaPrior& operator=(const EdgeCountDeltaPrior& other) {
        m_edgeCount = other.m_edgeCount;
        setState(m_edgeCount);
        return *this;
    }

    void sampleState() override { };
    const double getLogLikelihoodFromState(const size_t& state) const override { if (state == m_state) return 0.; else return -INFINITY; };
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { if (move.addedEdges.size() == move.removedEdges.size()) return 0; else return -INFINITY;}
    void checkSelfConsistency() const override { };

};

class EdgeCountPoissonPrior: public EdgeCountPrior{
private:
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

public:
    EdgeCountPoissonPrior() {}
    EdgeCountPoissonPrior(double mean) { setMean(mean); }
    EdgeCountPoissonPrior(const EdgeCountPoissonPrior& other) { setMean(other.m_mean); setState(other.m_state); }
    virtual ~EdgeCountPoissonPrior() {};
    const EdgeCountPoissonPrior& operator=(const EdgeCountPoissonPrior& other) {
        setMean(other.m_mean);
        setState(other.m_state);
        return *this;
    }

    double getMean() const { return m_mean; }
    void setMean(double mean){
        m_mean = mean;
        m_poissonDistribution = std::poisson_distribution<size_t>(mean);
    }
    void sampleState() override;
    const double getLogLikelihoodFromState(const size_t& state) const override;
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
