#ifndef FAST_MIDYNET_INTEGER_PARTITION_H
#define FAST_MIDYNET_INTEGER_PARTITION_H

#include "FastMIDyNet/types.h"
#include <iostream>

namespace FastMIDyNet{

std::vector<size_t> getConjugatePartition(std::list<size_t> partition);
std::vector<size_t> getCompactConjugatePartition(std::list<size_t> partition);
std::vector<size_t> getConjugatePartition(std::list<size_t> partition, size_t maxSize);
double q_rec(int n, int k);
double log_q_approx(size_t n, size_t k);
double log_q_approx_big(size_t n, size_t k);
double log_q_approx_small(size_t n, size_t k);

void printArray(std::vector<int> p);
void printAllRestrictedPartitions(int n, int m);

}
#endif // FAST_MIDYNET_INTEGER_PARTITION_H
