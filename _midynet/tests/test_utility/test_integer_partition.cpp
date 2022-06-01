#include "gtest/gtest.h"
#include "FastMIDyNet/utility/integer_partition.h"

namespace FastMIDyNet{

template<typename T>
void printContent(T container){
    std::cout << "{";
    size_t counter = 0;
    for (auto n : container){
        std::cout << n;
        ++counter;
        if (counter < container.size())
             std::cout << ", ";
    }
    std::cout << "}" << std::endl;
}

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

TEST(TestIntegerPartitionNumber, conjugatePartition_returnsCorrectConjugate){
    std::list<size_t> partition;
    std::vector<size_t> conjugate, compact;
    partition = {1,1,3,4};
    conjugate = {0,2,0,1,1,0,0,0,0,0};
    compact = {0,2,0,1,1};

    EXPECT_EQ(conjugate, getConjugatePartition(partition));
    EXPECT_EQ(compact, getCompactConjugatePartition(partition));

    partition = {0,9};
    conjugate = {1,0,0,0,0,0,0,0,0,1};
    compact = conjugate;
    EXPECT_EQ(conjugate, getConjugatePartition(partition));
    EXPECT_EQ(compact, getCompactConjugatePartition(partition));
}

}
