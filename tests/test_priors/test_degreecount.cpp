#include "gtest/gtest.h"

#include "FastMIDyNet/prior/dcsbm/degree.h"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility.h"


class DummyDegreeCountPrior: public FastMIDyNet::DegreeCountPrior {
    public:
        FastMIDyNet::DegreeSequence sample() { return {0}; }
        double getLogLikelihood(const FastMIDyNet::DegreeSequence& state) const { return 0; }

        void checkSelfConsistency() const {}
};

class TestEdgeCountPrior: public ::testing::Test {
    public:
        double poissonMean = 10;
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = FastMIDyNet::EdgeCountPoissonPrior(5);
        DummyDegreeCountPrior degreeCountPrior = DummyDegreeCountPrior(edgeCountPrior, poissonMean);
};


TEST_F(TestEdgeCountPrior, getLogLikelihoodRatio_blockMove_return0) {
    FastMIDyNet::BlockMove blockMove;
    EXPECT_EQ(degreeCountPrior.getLogLikelihoodRatio(blockMove), 0);
}

TEST_F(TestEdgeCountPrior, getLogPrior_anyState_returnHyperPriorLogLikelihood) {
    EXPECT_EQ(edgeCountPrior.getLogLikelihood(poissonMean), degreeCountPrior.getLogPrior());
}
