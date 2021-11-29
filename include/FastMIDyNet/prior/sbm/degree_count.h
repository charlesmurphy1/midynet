#ifndef FAST_MIDYNET_DEGREE_COUNT_H
#define FAST_MIDYNET_DEGREE_COUNT_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"


namespace FastMIDyNet{

class DegreeCountPrior: public Prior<DegreeSequence>{
protected:
    EdgeCountPrior& m_edgeCountPrior;
    size_t m_graphSize;

public:
    DegreeCountPrior(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        m_graphSize(graphSize), m_edgeCountPrior(edgeCountPrior) { }


    void samplePriors() override { m_edgeCountPrior.sample(); }
    double getLogLikelihood() const { return getLogLikelihoodFromState(m_state); }
    virtual double getLogLikelihoodFromState(const DegreeSequence&) const = 0;
    double getLogPrior() override { return m_edgeCountPrior.getLogJoint(); }
    virtual double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const = 0;
    double getLogJointRatio(const GraphMove& move) {
        return processRecursiveFunction<double>( [&]() {
                return getLogLikelihoodRatioFromGraphMove(move) + m_edgeCountPrior.getLogJointRatioFromGraphMove(move); },
                0);
    }
    double getLogJointRatio(const BlockMove& move) { return 0; }


    double getLogLikelihoodRatio(const BlockMove&) const { return 0; }
    void applyMove(const BlockMove&) { }
    void computationFinished() override { m_isProcessed = false; m_edgeCountPrior.computationFinished(); }
};


class DegreeCountUniformPrior: public DegreeCountPrior {
public:
    double getLogLikelihoodFromState(size_t state) const;
    double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    void applyGraphMove(const GraphMove&);

    void checkSelfConsistency() const;
};


}

#endif
