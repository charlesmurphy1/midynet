#ifndef FAST_MIDYNET_DEGREE_COUNT_H
#define FAST_MIDYNET_DEGREE_COUNT_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"


namespace FastMIDyNet{

class DegreeCountPrior: public Prior<DegreeSequence>{
    EdgeCountPrior& m_edgeCountPrior;
    size_t m_graphSize;

    public:
        DegreeCountPrior(size_t graphSize, EdgeCountPrior& edgeCountPrior):
            m_graphSize(graphSize), m_edgeCountPrior(edgeCountPrior) { }


        double getLogPrior() { return m_edgeCountPrior.getLogJoint(); }
        virtual double getLogLikelihoodRatio(const GraphMove& move) const = 0;
        double getLogJointRatio(const GraphMove& move) {
            double logJointRatio = 0;
            if (!m_isProcessed)
                logJointRatio = getLogLikelihoodRatio(move) + m_edgeCountPrior.getLogJointRatio(move);
            m_isProcessed = true;
            return logJointRatio;
        }
        double getLogJointRatio(const BlockMove& move) { return 0; }


        double getLogLikelihoodRatio(const BlockMove&) const { return 0; }
        void applyMove(const BlockMove&) { }
};


class DegreeCountUniformPrior: public DegreeCountPrior {
    public:
        double getLogLikelihood(size_t state) const;
        double getLogLikelihoodRatio(const GraphMove&) const;
        void applyMove(const GraphMove&);

        void checkSelfConsistency() const;
};


}

#endif
