#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/prior/dcsbm/degree_count.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

class DegreePrior: public Prior<DegreeSequence>{

    public:

        virtual double getLogPrior() = 0;
        double getLogJointRatio(const MultiBlockMove& move) { return 0; }
        double getLogJointRatio(const GraphMove& move) { return 0; }

        double getLogLikelihoodRatio(const MultiBlockMove&) const { return 0; }
        double getLogLikelihoodRatio(const GraphMove&) const { return 0; }
        void applyMove(const BlockMove&) { }
};


}

#endif
