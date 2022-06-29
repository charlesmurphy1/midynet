#ifndef FAST_MIDYNET_RNG_H
#define FAST_MIDYNET_RNG_H


#include <random>
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {

extern RNG rng;
extern size_t SEED;

void seed(size_t n);
void seedWithTime();
const size_t& getSeed();

} // namespace FastMIDyNet

#endif
