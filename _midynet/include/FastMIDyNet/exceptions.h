#ifndef FAST_MIDYNET_EXCEPTIONS_H
#define FAST_MIDYNET_EXCEPTIONS_H


#include <stdexcept>
#include <string>


namespace FastMIDyNet {

void assertValidProbability(double probability);

class ConsistencyError: public std::logic_error {
public:
    ConsistencyError(const std::string& message): std::logic_error(message) {}
};

class SafetyError: public std::logic_error {
public:
    SafetyError(const std::string& message): std::logic_error(message) {}
};

} // namespace FastMIDyNet

#endif
