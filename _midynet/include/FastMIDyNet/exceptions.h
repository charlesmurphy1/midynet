#ifndef FAST_MIDYNET_EXCEPTIONS_H
#define FAST_MIDYNET_EXCEPTIONS_H


#include <stdexcept>
#include <string>


namespace FastMIDyNet {

void assertValidProbability(double probability);

class ConsistencyError: public std::runtime_error {
public:
    // Custom message
    ConsistencyError(const std::string& message): std::runtime_error(message) {}

    // Message without values
    ConsistencyError(
        const std::string& className,
        const std::string& expectedProperty,
        const std::string& actualProperty
    ): std::runtime_error(
        className + ": `" + expectedProperty + "` is inconsistent with `" + actualProperty + "`."
    ) {}

    // Message with values
    ConsistencyError(
        const std::string& className,
        const std::string& expectedProperty,
        const std::string& expectedValue,
        const std::string& actualProperty,
        const std::string& actualValue
    ): std::runtime_error(
        className + ": `" + expectedProperty + "` (value=`" + expectedValue + "`) is inconsistent with `"
        + actualProperty + "` (value=`" + actualValue + "`)."
    ) {}

    // Message with values and locations
    ConsistencyError(
        const std::string& className,
        const std::string& expectedName,
        const std::string& expectedValue,
        const std::string& actualName,
        const std::string& actualValue,
        const std::string& location
    ): std::runtime_error(
        className + ": `" + expectedName + "` (" + expectedValue + ") is inconsistent with `"
        + actualName + "` (" + actualValue + ") at [" + location + "]."
    ) {}
};

class SafetyError: public std::runtime_error {
public:
    // Custom message
    SafetyError(const std::string& message): std::runtime_error(message) {}

    // Standard message
    SafetyError(
        const std::string& className,
        const std::string& variableName,
        const std::string& value="nullptr"
    ): std::runtime_error(
        className + ": unsafe `" + variableName + "` with value `" + value + "`."
    ) {}
};

class DepletedMethodError: public std::runtime_error {
public:
    // Custom message
    DepletedMethodError(const std::string& message): std::runtime_error(message) {}

    // Standard message
    DepletedMethodError(
        const std::string& className,
        const std::string& depletedMethodName
    ): std::runtime_error(
        className + ": method `" + depletedMethodName + "` is depleted and should not be used."
    ) {}


    // Standard message with "use instead"
    DepletedMethodError(
        const std::string& className,
        const std::string& depletedMethodName,
        const std::string& correctMethodName
    ): std::runtime_error(
        className + ": method `" + depletedMethodName
        + "` is depleted and should not be used; use instead `"
        + correctMethodName + "`."
    ) {}
};

} // namespace FastMIDyNet

#endif
