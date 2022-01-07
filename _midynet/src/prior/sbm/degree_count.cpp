#include "FastMIDyNet/prior/sbm/degree_count.h"


using namespace FastMIDyNet;



double DegreeCountUniformPrior::getLogLikelihoodFromState(size_t state) const{
    return 0.;
}
double DegreeCountUniformPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove&) const{
    return 0.;
}
void DegreeCountUniformPrior::applyGraphMove(const GraphMove&){

}

void DegreeCountUniformPrior::checkSelfConsistency() const{

}
