#ifndef FAST_MIDYNET_UTILITY_H
#define FAST_MIDYNET_UTILITY_H


#include <random>
#include <string>
#include <stdexcept>
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {

extern RNG rng;

void setSeed(size_t seed) ;

} // namespace FastMIDyNet

#endif
