#include <chrono>
#include <random>
#include <iostream>

#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {

RNG rng;
size_t SEED=0;
void seed(size_t seed){ SEED=seed; rng.seed(seed); std::srand(seed); }
void seedWithTime(){
    seed(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
const size_t& getSeed() { return SEED; }

}
