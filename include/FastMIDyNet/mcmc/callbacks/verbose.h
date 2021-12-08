#ifndef FAST_MIDYNET_VERBOSE_H
#define FAST_MIDYNET_VERBOSE_H

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/utility/functions.h"
#include "callback.h"

namespace FastMIDyNet{

// class Verbose: public CallBack{
// private:
//     MCMC* m_mcmcPtr;
// public:
//     void setUp(MCMC* mcmcPtr){ m_mcmcPtr = mcmcPtr; }
//     virtual std::string getMessage() = 0;
//     virtual std::string update() = 0;
// };
//
// class StepVerbose: public Verbose{
// private:
//     size_t m_numberOfSuccess, m_numberOfFailure;
//     size_t m_totalNumberOfSuccess, m_totalNumberOfFailure;
// public:
//
//
// };

class Verbose: public CallBack{
protected:
    std::string name;
public:
    const std::string& getName() const { return name; }
    virtual std::string getMessage() const = 0;
};

class TimerVerbose: public Verbose{
protected:
    std::string name = "Time";
    std::chrono::time_point<std::chrono::steady_clock> m_start, m_end;
public:
    std::string getMessage() const{
        std::stringstream message;
        message.precision(4);
        message << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count();
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
    std::string name = "Success";
    size_t m_numberOfSuccesses;
public:
    void onSweepBegin(){ m_numberOfSuccesses = 0; }
    void onStepEnd(){ if ( m_mcmcPtr->isLastAccepted() ) ++m_numberOfSuccesses; }
    std::string getMessage() const{
        std::stringstream message;
        message.precision(4);
        message << "Failure: " << m_numberOfSuccesses;
        return message.str();
    }
};

class FailureCounterVerbose: public Verbose{
protected:
    std::string name = "Failure";
    size_t m_numberOfFailure;
public:
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
    virtual double updateSavedRatio() const = 0;
    void onStepEnd(){ m_savedLogJointRatio = updateSavedRatio(); }
    std::string getMessage() const{
        std::stringstream message;
        message.precision(6);
        message << name << ": " << m_savedLogJointRatio;
        return message.str();
    }
};

class MinimumLogJointRatioVerbose: public LogJointRatioVerbose{
protected:
    std::string name = "min(dS)";
public:
    void onBegin(){ m_savedLogJointRatio = INFINITY; }
    double updateSavedRatio() const {
        if (m_mcmcPtr->getLastLogJointRatio() < m_savedLogJointRatio)
            return m_mcmcPtr->getLastLogJointRatio();
        else
            return m_savedLogJointRatio;
    }
};

class MaximumLogJointRatioVerbose: public LogJointRatioVerbose{
protected:
    std::string name = "max(dS)";
public:
    void onBegin(){ m_savedLogJointRatio = -INFINITY; }
    double updateSavedRatio() const {
        if (m_mcmcPtr->getLastLogJointRatio() > m_savedLogJointRatio)
            return m_mcmcPtr->getLastLogJointRatio();
        else
            return m_savedLogJointRatio;
    }
};

class MeanLogJointRatioVerbose: public LogJointRatioVerbose{
protected:
    std::string name = "mean(dS)";
public:
    void onBegin(){ m_savedLogJointRatio = 0; }
    double updateSavedRatio() const {
        size_t numSteps = m_mcmcPtr->getNumSteps();
        return (m_savedLogJointRatio * (numSteps - 1) + m_mcmcPtr->getLastLogJointRatio()) / (numSteps);
    }
};

class VerboseDisplay: public CallBack{
protected:
    std::vector<Verbose*> m_verboseVec;
public:
    VerboseDisplay(std::vector<Verbose*> verboseVec={}): m_verboseVec(verboseVec){}

    std::string getMessage() const;
    virtual void writeMessage(std::string message) = 0;
    void writeMessage() { writeMessage(getMessage()); }
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
