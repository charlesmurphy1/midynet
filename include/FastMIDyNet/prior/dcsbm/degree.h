#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/prior/dcsbm/degree_count.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

class DegreePrior: public Prior<DegreeSequence>{
protected:
    size_t m_graphSize;
public:

    DegreePrior(size_t graphSize): m_graphSize(graphSize) {}

    size_t getGraphSize() { return m_graphSize; }

    double getLogLikelihoodRatio(const BlockMove&) const { return 0; }
    double getLogLikelihoodRatio(const GraphMove&) const { return 0; }

    double getLogJointRatio(const BlockMove& move) { return 0; }
    double getLogJointRatio(const GraphMove& move) { return 0; }

    double getLogPriorRatio(const BlockMove& move) { return 0; }
    double getLogPriorRatio(const GraphMove& move) { return 0; }

    double getLogPrior() { return 0;};

    void applyMove(const BlockMove&) { }
    void applyMove(const GraphMove&) { }
};


}

#endif
