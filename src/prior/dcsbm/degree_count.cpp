#include "FastMIDyNet/prior/dcsbm/degree_count.h"


using namespace FastMIDyNet;



double DegreeCountUniformPrior::getLogLikelihood(size_t state) const{
    return 0.;
}
double DegreeCountUniformPrior::getLogLikelihoodRatio(const GraphMove&) const{
    return 0.;
}
void DegreeCountUniformPrior::applyMove(const GraphMove&){

}

void DegreeCountUniformPrior::checkSelfConsistency() const{

}
