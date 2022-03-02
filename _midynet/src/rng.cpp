#include <chrono>
#include <random>
#include <iostream>

#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {

RNG rng;

void seed(size_t seed){ rng.seed(seed); std::srand(seed); }
void seedWithTime(){
    seed(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

}
