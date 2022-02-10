#include "gtest/gtest.h"
#include "FastMIDyNet/utility/integer_partition.h"

namespace FastMIDyNet{

TEST(TestIntegerPartitionNumber, q_rec_recursiveExpression){
    for (size_t i=2; i<100; ++i)
        EXPECT_EQ(q_rec(i, 1), 1);
    EXPECT_EQ(q_rec(5, 1), 1);
    EXPECT_EQ(q_rec(5, 2), 3);
    EXPECT_EQ(q_rec(5, 3), 5);
    EXPECT_EQ(q_rec(5, 4), 6);
    EXPECT_EQ(q_rec(5, 5), 7);
}

TEST(TestIntegerPartitionNumber, log_q_approx_returnResult){
    size_t m = 5, n = 50;
    double exact = log(q_rec(n, m));
    double approx = log_q_approx(n, m);
    EXPECT_NEAR(exact, approx, 1);
}

}
