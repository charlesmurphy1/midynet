#include "FastMIDyNet/mcmc/mcmc.h"
#include <iostream>
#include <stdexcept>

namespace FastMIDyNet{

std::tuple<size_t, size_t> MCMC::doMHSweep(size_t burn){
    onSweepBegin();
    size_t numSuccess = 0, numFailure = 0;
    for (size_t i = 0; i < burn; i++) {
        bool isAccepted = doMetropolisHastingsStep();
        if (isAccepted) ++numSuccess;
        else ++numFailure;
        ++m_numSteps;
    }
    onSweepEnd();
    ++m_numSweeps;

    return {numSuccess, numFailure};
}

}
