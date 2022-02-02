#include "FastMIDyNet/mcmc/callbacks/verbose.h"

namespace FastMIDyNet{

std::string VerboseDisplay::getMessage() const {
    std::stringstream message;
    message << "Sweep " << m_mcmcPtr->getNumSweeps() << ": ";
    for (auto v : m_verboseVec){
        message << v->getMessage() << " ";
    }
    return message.str();
}
void VerboseDisplay::setUp(MCMC* mcmcPtr) {
    CallBack::setUp(mcmcPtr);
    for (auto v : m_verboseVec) v->setUp(mcmcPtr);
}
void VerboseDisplay::tearDown() {
    for (auto v : m_verboseVec) v->tearDown();
}
void VerboseDisplay::onBegin() {
    for (auto v : m_verboseVec) v->onBegin();
}
void VerboseDisplay::onEnd() {
    for (auto v : m_verboseVec) v->onEnd();
}
void VerboseDisplay::onStepBegin() {
    for (auto v : m_verboseVec) v->onStepBegin();
}
void VerboseDisplay::onStepEnd() {
    for (auto v : m_verboseVec) v->onStepEnd();
}
void VerboseDisplay::onSweepBegin() {
    for (auto v : m_verboseVec) v->onSweepBegin();
}
void VerboseDisplay::onSweepEnd() {
    for (auto v : m_verboseVec) v->onSweepEnd();
    if (m_mcmcPtr->getNumSweeps() % m_step == 0)
        writeMessage();
}

}
