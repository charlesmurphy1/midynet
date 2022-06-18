#ifndef FAST_MIDYNET_VERBOSE_H
#define FAST_MIDYNET_VERBOSE_H

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <math.h>
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/utility/functions.h"
#include "callback.h"

namespace FastMIDyNet{

class Verbose: public CallBack<MCMC>{
protected:
    std::string m_name;
public:
    Verbose(std::string name="Verbose"):m_name(name){}
    const std::string& getName() const { return m_name; }
    virtual std::string getMessage() const = 0;
};

class TimerVerbose: public Verbose{
protected:
    std::chrono::time_point<std::chrono::steady_clock> m_start, m_end;
public:
    TimerVerbose():Verbose("Time"){};
    std::string getMessage() const{
        std::stringstream message;
        message.precision(4);
        message << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count() << " ms";
        return message.str();
    }
    void onSweepBegin() {
        m_start = std::chrono::steady_clock::now();
    }
    void onSweepEnd() {
        m_end = std::chrono::steady_clock::now();
    }
};

class SuccessCounterVerbose: public Verbose{
protected:
    size_t m_numberOfSuccesses;
public:
    SuccessCounterVerbose():Verbose("Success"){};
    void onSweepBegin(){ m_numberOfSuccesses = 0; }
    void onStepEnd(){ if ( m_mcmcPtr->isLastAccepted() ) ++m_numberOfSuccesses; }
    std::string getMessage() const{
        std::stringstream message;
        message.precision(4);
        message << "Success: " << m_numberOfSuccesses;
        return message.str();
    }
};

class FailureCounterVerbose: public Verbose{
protected:
    size_t m_numberOfFailure;
public:
    FailureCounterVerbose():Verbose("Failure"){};
    void onSweepBegin(){ m_numberOfFailure = 0; }
    void onStepEnd(){ if ( not m_mcmcPtr->isLastAccepted() ) ++m_numberOfFailure; }
    std::string getMessage() const{
        std::stringstream message;
        message.precision(4);
        message << "Failure: " << m_numberOfFailure;
        return message.str();
    }
};

class LogJointRatioVerbose: public Verbose{
protected:
    double m_savedLogJointRatio;
public:
    LogJointRatioVerbose(std::string name="dS"):Verbose(name){};
    virtual double updateSaved() const = 0;
    void onStepEnd(){
        m_savedLogJointRatio = updateSaved();
    }
    std::string getMessage() const{
        std::stringstream message;
        message.precision(6);
        message << m_name << ": " << m_savedLogJointRatio;
        return message.str();
    }
};

class MinimumLogJointRatioVerbose: public LogJointRatioVerbose{
public:
    MinimumLogJointRatioVerbose():LogJointRatioVerbose("min(dS)"){};
    void onBegin(){ m_savedLogJointRatio = INFINITY; }
    double updateSaved() const {
        if (m_mcmcPtr->getLastLogJointRatio() < m_savedLogJointRatio and
            m_mcmcPtr->getLastLogJointRatio() != -INFINITY and
            m_mcmcPtr->getLastLogJointRatio() != INFINITY)
            return m_mcmcPtr->getLastLogJointRatio();
        else
            return m_savedLogJointRatio;
    }
};

class MaximumLogJointRatioVerbose: public LogJointRatioVerbose{
public:
    MaximumLogJointRatioVerbose():LogJointRatioVerbose("max(dS)"){};
    void onBegin(){ m_savedLogJointRatio = -INFINITY; }
    double updateSaved() const {
        if (m_mcmcPtr->getLastLogJointRatio() > m_savedLogJointRatio and
            m_mcmcPtr->getLastLogJointRatio() != -INFINITY and
            m_mcmcPtr->getLastLogJointRatio() != INFINITY)
            return m_mcmcPtr->getLastLogJointRatio();
        else
            return m_savedLogJointRatio;
    }
};

class MeanLogJointRatioVerbose: public LogJointRatioVerbose{
public:
    MeanLogJointRatioVerbose():LogJointRatioVerbose("mean(dS)"){};
    void onBegin(){ m_savedLogJointRatio = 0; }
    double updateSaved() const {
        size_t numSteps = m_mcmcPtr->getNumSteps();
        if (m_mcmcPtr->getLastLogJointRatio() == -INFINITY or
            m_mcmcPtr->getLastLogJointRatio() == INFINITY)
            return m_savedLogJointRatio;
        double newMean = (m_savedLogJointRatio * (numSteps - 1) + m_mcmcPtr->getLastLogJointRatio()) / (numSteps);
        if ( isnan(newMean) )
            return m_savedLogJointRatio;
        else
            return newMean;
    }
};

class MinimumLogAcceptationVerbose: public LogJointRatioVerbose{
public:
    MinimumLogAcceptationVerbose():LogJointRatioVerbose("min(acc_p)"){};
    void onBegin(){ m_savedLogJointRatio = INFINITY; }
    double updateSaved() const {
        if (m_mcmcPtr->getLastLogAcceptance() < m_savedLogJointRatio and
            m_mcmcPtr->getLastLogAcceptance() != -INFINITY and
            m_mcmcPtr->getLastLogAcceptance() != INFINITY)
            return m_mcmcPtr->getLastLogAcceptance();
        else
            return m_savedLogJointRatio;
    }
};

class MaximumLogAcceptationVerbose: public LogJointRatioVerbose{
public:
    MaximumLogAcceptationVerbose():LogJointRatioVerbose("max(acc_p)"){};
    void onBegin(){ m_savedLogJointRatio = -INFINITY; }
    double updateSaved() const {
        if (m_mcmcPtr->getLastLogAcceptance() > m_savedLogJointRatio and
            m_mcmcPtr->getLastLogAcceptance() != -INFINITY and
            m_mcmcPtr->getLastLogAcceptance() != INFINITY)
            return m_mcmcPtr->getLastLogAcceptance();
        else
            return m_savedLogJointRatio;
    }
};

class MeanLogAcceptationVerbose: public LogJointRatioVerbose{
public:
    MeanLogAcceptationVerbose():LogJointRatioVerbose("mean(acc_p)"){};
    void onBegin(){ m_savedLogJointRatio = 0; }
    double updateSaved() const {
        size_t numSteps = m_mcmcPtr->getNumSteps();
        if (m_mcmcPtr->getLastLogAcceptance() == -INFINITY or
            m_mcmcPtr->getLastLogAcceptance() == INFINITY)
            return m_savedLogJointRatio;
        double newMean = (m_savedLogJointRatio * (numSteps - 1) + m_mcmcPtr->getLastLogAcceptance()) / (numSteps);
        if ( isnan(newMean) )
            return m_savedLogJointRatio;
        else
            return newMean;
    }
};

class VerboseDisplay: public CallBack<MCMC>{
protected:
    std::vector<Verbose*> m_verboseVec;
    size_t m_step;
    size_t m_numSweeps;
public:
    VerboseDisplay(std::vector<Verbose*> verboseVec={}): m_verboseVec(verboseVec), m_step(1){}

    std::string getMessage() const;
    virtual void writeMessage(std::string message) = 0;
    void writeMessage() { writeMessage(getMessage()); }
    size_t getStep() { return m_step; }
    void setStep(size_t step) { m_step = step; }
    void setUp(MCMC* mcmcPtr) ;
    void tearDown() ;
    virtual void onBegin() ;
    virtual void onEnd() ;
    void onStepBegin() ;
    void onStepEnd() ;
    void onSweepBegin() ;
    void onSweepEnd() ;
};

class VerboseToConsole: public VerboseDisplay{
public:
    using VerboseDisplay::VerboseDisplay;
    void writeMessage(std::string message) { std::cout << getMessage() << std::endl; }
};

class VerboseToFile: public VerboseDisplay{
private:
    std::string m_filename;
    std::ofstream m_file;
public:
    VerboseToFile(std::string filename="verbose", std::vector<Verbose*> verboseVec={}):
    m_filename(filename), VerboseDisplay(verboseVec){}

    void onBegin() { m_file.open(m_filename); }
    void onEnd() { m_file.close(); }
    void writeMessage(std::string message) { m_file << message << std::endl; }
};


}

#endif
