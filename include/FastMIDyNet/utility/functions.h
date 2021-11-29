#ifndef FAST_MIDYNET_UTIL_FUNCTIONS_H
#define FAST_MIDYNET_UTIL_FUNCTIONS_H

#include <list>
#include <utility>
#include <iostream>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

namespace FastMIDyNet {

// static const double INFINITY = std::numeric_limits<double>::infinity();

double logFactorial(size_t);
double logDoubleFactorial(size_t);
double logBinomialCoefficient(size_t, size_t);
double logPoissonPMF(size_t x, double mean);
double logMultinomialCoefficient(std::list<size_t> sequence);
double logMultisetCoefficient(size_t n, size_t k);

double logRestrictedPartitionNumber(size_t n, size_t k);
double logRestrictedPartitionNumber(size_t n, size_t k);
double logApproxRestrictedPartitionNumber(size_t n, size_t k);

template<typename T>
std::pair<T, T> getOrderedPair(const std::pair<T, T>& myPair){
    if (myPair.first < myPair.second)
        return myPair;
    return {myPair.second, myPair.first};
}
BaseGraph::Edge getOrderedEdge(const BaseGraph::Edge&);
inline size_t choose2(size_t n) { return n*(n-1)/2; }
std::pair<size_t, size_t> getUndirectedPairFromIndex(size_t index, size_t n);

template<typename T>
T sumElementsOfMatrix(Matrix<T> mat, T init){
    T sum = init;
    for (auto rows = mat.begin(); rows != mat.end(); rows ++){
        for (auto cols = rows->begin(); cols != rows->end(); cols ++){
            sum += *cols;
        }
    }
    return sum;
}

template<typename T>
static void verifyVectorHasSize(
    const std::vector<T>& vec,
    size_t size,
    const std::string& vectorName,
    const std::string& sizeName) {
    if (vec.size() != size)
        throw ConsistencyError("EdgeMatrixPrior: "+vectorName+" has size "+
                std::to_string(vec.size())+" while there are "+
                std::to_string(size)+" "+sizeName+".");
}

template<typename T>
void displayMatrix(Matrix<T> matrix, std::string name="m"){
    std::cout << name << " = [" << std::endl;
    for (auto row : matrix){
        std::cout << "  [ ";
        for (auto col : row){
            std::cout << std::to_string(col) << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}
template<typename T>
void displayVector(std::vector<T> vec, std::string name="v"){
    std::cout << name << " = [ " << std::endl;
    for (auto row : vec){
        std::cout << std::to_string(row) << " ";
    }
    std::cout << "]" << std::endl;
}


} // namespace FastMIDyNet

#endif
