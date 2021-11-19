#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"


namespace FastMIDyNet {


double EdgeMatrixUniformPrior::getLogLikelihoodRatio(const GraphMove& move) const {
    return 0.;
};

double EdgeMatrixUniformPrior::getLogLikelihoodRatio(const BlockMove& move) const {
    return 0.;
}


void EdgeMatrixUniformPrior::applyMove(const GraphMove& move) {
}

void EdgeMatrixUniformPrior::applyMove(const BlockMove& move) {
}


} // namespace FastMIDyNet
