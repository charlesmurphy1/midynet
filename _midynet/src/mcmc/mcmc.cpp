#include "FastMIDyNet/mcmc/mcmc.h"
#include <iostream>
#include <stdexcept>

namespace FastMIDyNet{

void MCMC::doMHSweep(size_t burn){
    if (not m_hasState){
        throw std::logic_error("MCMC: state is empty, cannot do sweep.");
    }
    m_callBacks.onSweepBegin();
    for (size_t i = 0; i < burn; i++) {
        m_callBacks.onStepBegin();

        doMetropolisHastingsStep();
        ++m_numSteps;

        m_callBacks.onStepEnd();
    }
    m_callBacks.onSweepEnd();
    ++m_numSweeps;
}

}