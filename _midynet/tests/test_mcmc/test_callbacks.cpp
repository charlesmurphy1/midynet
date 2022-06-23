#include "gtest/gtest.h"

#include "fixtures.hpp"
#include "FastMIDyNet/dynamics/sis.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/proposer/edge/hinge_flip.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "FastMIDyNet/mcmc/callbacks/collector.hpp"
#include "FastMIDyNet/mcmc/callbacks/verbose.h"
namespace FastMIDyNet{

#define CALLBACK_TESTS(CALLBACK_TESTSUITE)\
    TEST_F(CALLBACK_TESTSUITE, onBegin_raiseNoExceptionOrSegFault){\
        callback.onBegin();\
    }\
    TEST_F(CALLBACK_TESTSUITE, onEnd_raiseNoExceptionOrSegFault){\
        callback.onEnd();\
    }\
    TEST_F(CALLBACK_TESTSUITE, onStepBegin_raiseNoExceptionOrSegFault){\
        callback.onStepBegin();\
    }\
    TEST_F(CALLBACK_TESTSUITE, onStepEnd_raiseNoExceptionOrSegFault){\
        callback.onStepEnd();\
    }\
    TEST_F(CALLBACK_TESTSUITE, onSweepBegin_raiseNoExceptionOrSegFault){\
        callback.onSweepBegin();\
    }\
    TEST_F(CALLBACK_TESTSUITE, onSweepEnd_raiseNoExceptionOrSegFault){\
        callback.onSweepEnd();\
    }\

#define COLLECTOR_TESTS(CALLBACK_TESTSUITE)\
    CALLBACK_TESTS(CALLBACK_TESTSUITE);\
    TEST_F(CALLBACK_TESTSUITE, collect){\
        callback.collect();\
    }\


class TestCallBackBaseClass: public::testing::Test{
public:
    CallBack<MCMC> callback ;
    DummyMCMC mcmc;
    std::string name = "generic";
    void SetUp(){
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};

CALLBACK_TESTS(TestCallBackBaseClass);

class TestCollectGraphOnSweep: public::testing::Test{
public:
    CollectGraphOnSweep<GraphReconstructionMCMC<RandomGraph>> callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "collect_graph";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
COLLECTOR_TESTS(TestCollectGraphOnSweep);

class TestCollectEdgeMultiplicityOnSweep: public::testing::Test{
public:
    CollectEdgeMultiplicityOnSweep<GraphReconstructionMCMC<RandomGraph>> callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "collect_edge_multiplicity";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
COLLECTOR_TESTS(TestCollectEdgeMultiplicityOnSweep);

class TestCollectLikelihoodOnSweep: public::testing::Test{
public:
    CollectLikelihoodOnSweep callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "collect_likelihood";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
COLLECTOR_TESTS(TestCollectLikelihoodOnSweep);

class TestCollectPriorOnSweep: public::testing::Test{
public:
    CollectPriorOnSweep callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "collect_prior";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
COLLECTOR_TESTS(TestCollectPriorOnSweep);

class TestCollectJointOnSweep: public::testing::Test{
public:
    CollectJointOnSweep callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "collect_joint";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
COLLECTOR_TESTS(TestCollectJointOnSweep);

class TestTimerVerbose: public::testing::Test{
public:
    TimerVerbose callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "timer";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
CALLBACK_TESTS(TestTimerVerbose);

class TestSuccessCounterVerbose: public::testing::Test{
public:
    SuccessCounterVerbose callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "success_counter";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
CALLBACK_TESTS(TestSuccessCounterVerbose);

class TestFailureCounterVerbose: public::testing::Test{
public:
    FailureCounterVerbose callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "failure_counter";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
CALLBACK_TESTS(TestFailureCounterVerbose);


class TestMeanLogJointRatioVerbose: public::testing::Test{
public:
    MeanLogJointRatioVerbose callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "mean_joint_ratio";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
CALLBACK_TESTS(TestMeanLogJointRatioVerbose);


class TestMaximumLogJointRatioVerbose: public::testing::Test{
public:
    MaximumLogJointRatioVerbose callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "max_joint_ratio";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
CALLBACK_TESTS(TestMaximumLogJointRatioVerbose);


class TestMinimumLogJointRatioVerbose: public::testing::Test{
public:
    MinimumLogJointRatioVerbose callback ;
    DummyGraphPrior randomGraph = DummyGraphPrior();
    SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);
    HingeFlipUniformProposer proposer = HingeFlipUniformProposer();
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, proposer);
    std::string name = "min_joint_ratio";
    void SetUp(){
        dynamics.sample();
        mcmc.insertCallBack(name, callback);
        mcmc.setUp();
    }
};
CALLBACK_TESTS(TestMinimumLogJointRatioVerbose);

}
