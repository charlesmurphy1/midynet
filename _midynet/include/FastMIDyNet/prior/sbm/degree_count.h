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
    const double getLogLikelihood() const override { return getLogLikelihoodFromState(m_state); }
    virtual const double getLogLikelihoodFromState(const DegreeSequence&) const = 0;
    const double getLogPrior() const override { return m_edgeCountPrior.getLogJoint(); }
    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const = 0;
    const double getLogJointRatio(const GraphMove& move) {
        return processRecursiveFunction<double>( [&]() {
                return getLogLikelihoodRatioFromGraphMove(move) + m_edgeCountPrior.getLogJointRatioFromGraphMove(move); },
                0);
    }
    const double getLogJointRatio(const BlockMove& move) { return 0; }


    const double getLogLikelihoodRatio(const BlockMove&) const { return 0; }
    void applyMove(const BlockMove&) { }
    void computationFinished() const override { m_isProcessed = false; m_edgeCountPrior.computationFinished(); }
    virtual void checkSafety() const override{ }
};


class DegreeCountUniformPrior: public DegreeCountPrior {
public:
    const double getLogLikelihoodFromState(const DegreeSequence& state) const override;
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override;
    void applyGraphMove(const GraphMove&);

    void checkSelfConsistency() const;
};


}

#endif
