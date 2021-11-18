#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"


namespace FastMIDyNet{

class DegreeCountPrior: public Prior<DegreeSequence>{
    const EdgeCountPrior& m_edgeCountPrior;
    size_t m_graphSize;

    public:
        DegreeCountPrior(const EdgeCountPrior& edgeCountPrior, size_t graphSize):
            m_edgeCountPrior(edgeCountPrior), m_graphSize(graphSize) { }

        double getLogLikelihoodRatio(const BlockMove&) const { return 0; }
        void applyMove(const BlockMove&) { }
        double getLogPrior() const { return m_edgeCountPrior.getLogLikelihood(); }
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
