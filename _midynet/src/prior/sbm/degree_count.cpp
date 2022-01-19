#include "FastMIDyNet/prior/sbm/degree_count.h"


using namespace FastMIDyNet;



const double DegreeCountUniformPrior::getLogLikelihoodFromState(const DegreeSequence& state) const{
    return 0.;
}
const double DegreeCountUniformPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove&) const{
    return 0.;
}
void DegreeCountUniformPrior::applyGraphMove(const GraphMove&){

}

void DegreeCountUniformPrior::checkSelfConsistency() const{

}
