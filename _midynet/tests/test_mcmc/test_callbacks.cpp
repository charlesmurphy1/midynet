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

#define CALLBACK_TESTS(TEST_CALL_BACK, TESTED_CALL_CLASS)\
    class TEST_CALL_BACK: public::testing::Test{\
    public:\
        using BaseClass = GraphReconstructionMCMC<RandomGraph>;\
        DummyER randomGraph = DummyER();\
        SISDynamics<RandomGraph> dynamics = SISDynamics<RandomGraph>(randomGraph, 10, 0.1);\
        HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();\
        GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics, edgeProposer);\
        TESTED_CALL_CLASS callback = TESTED_CALL_CLASS();\
        void SetUp(){\
            seedWithTime();\
            mcmc.insertCallBack("generic", callback);\
            mcmc.setUp();\
        }\
        void TearDown(){\
            mcmc.tearDown();\
            mcmc.removeCallBack("generic");\
        }\
    };\
    TEST_F(TEST_CALL_BACK, tearDown_raiseNoExceptionOrSegFault){\
        callback.tearDown();\
    }\
    TEST_F(TEST_CALL_BACK, onBegin_raiseNoExceptionOrSegFault){\
        callback.onBegin();\
    }\
    TEST_F(TEST_CALL_BACK, onEnd_raiseNoExceptionOrSegFault){\
        callback.onEnd();\
    }\
    TEST_F(TEST_CALL_BACK, onStepBegin_raiseNoExceptionOrSegFault){\
        callback.onStepBegin();\
    }\
    TEST_F(TEST_CALL_BACK, onStepEnd_raiseNoExceptionOrSegFault){\
        callback.onStepEnd();\
    }\
    TEST_F(TEST_CALL_BACK, onSweepBegin_raiseNoExceptionOrSegFault){\
        callback.onSweepBegin();\
    }\
    TEST_F(TEST_CALL_BACK, onSweepEnd_raiseNoExceptionOrSegFault){\
        callback.onSweepEnd();\
    }\

#define COLLECTOR_TESTS(TEST_COLLECTOR, TESTED_COLLECTOR_CLASS, GETTER)\
    CALLBACK_TESTS(TEST_COLLECTOR, TESTED_COLLECTOR_CLASS);\
    TEST_F(TEST_COLLECTOR, onSweepEnd_collect_expectVectorNotEmpty){\
        callback.onSweepEnd();\
        EXPECT_EQ(callback.GETTER().size(), 1);\
    }\


CALLBACK_TESTS(TestCallBackBaseClass, CallBack<MCMC>);

COLLECTOR_TESTS(TestCollectGraphOnSweep, CollectGraphOnSweep<BaseClass>, getGraphs);

CALLBACK_TESTS(TestCollectEdgeMultiplicityOnSweep, CollectEdgeMultiplicityOnSweep<BaseClass>);

// COLLECTOR_TESTS(TestCollectPartitionOnSweep, CollectPartitionOnSweep<BaseClass>, getPartitions);

COLLECTOR_TESTS(TestCollectLikelihoodOnSweep, CollectLikelihoodOnSweep, getLogLikelihoods);

COLLECTOR_TESTS(TestCollectPriorOnSweep, CollectPriorOnSweep, getLogPriors);

COLLECTOR_TESTS(TestCollectJointOnSweep, CollectJointOnSweep, getLogJoints);

CALLBACK_TESTS(TestTimerVerbose, TimerVerbose);

CALLBACK_TESTS(TestuccessCounterVerbose, SuccessCounterVerbose);

CALLBACK_TESTS(TestFailureCounterVerbose, FailureCounterVerbose);

CALLBACK_TESTS(TestMinimumLogJointRatioVerbose, MinimumLogJointRatioVerbose);

CALLBACK_TESTS(TestMaximumLogJointRatioVerbose, MaximumLogJointRatioVerbose);

CALLBACK_TESTS(TestMeanLogJointRatioVerbose, MeanLogJointRatioVerbose);

}
