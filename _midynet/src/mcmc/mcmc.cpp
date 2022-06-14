#include "FastMIDyNet/mcmc/mcmc.h"
#include <iostream>
#include <stdexcept>

namespace FastMIDyNet{

std::pair<size_t, size_t> MCMC::doMHSweep(size_t burn, bool checkingConsistency, bool checkingSafety){
    m_callBacks.onSweepBegin();
    size_t numSuccess = 0, numFailure = 0;
    for (size_t i = 0; i < burn; i++) {
        m_callBacks.onStepBegin();
        bool isAccepted = doMetropolisHastingsStep();
        if (isAccepted) ++numSuccess;
        else ++numFailure;
        ++m_numSteps;

        m_callBacks.onStepEnd();
        if (checkingConsistency)
            checkConsistency();
        if (checkingSafety)
            checkSafety();
    }
    m_callBacks.onSweepEnd();
    ++m_numSweeps;

    return {numSuccess, numFailure};
}

}
