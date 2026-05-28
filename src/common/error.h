#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

namespace cann {

enum class Status : uint8_t {
    SUCCESS = 0,
    INVALID_PARAM = 1,
    INVALID_RANK = 2,
    INVALID_DATATYPE = 3,
    INTERNAL_ERROR = 4,
    TIMEOUT = 5,
    LINK_FAILURE = 6,
};

inline const char* StatusToString(Status s) {
    switch (s) {
        case Status::SUCCESS:          return "SUCCESS";
        case Status::INVALID_PARAM:    return "INVALID_PARAM";
        case Status::INVALID_RANK:     return "INVALID_RANK";
        case Status::INVALID_DATATYPE: return "INVALID_DATATYPE";
        case Status::INTERNAL_ERROR:   return "INTERNAL_ERROR";
        case Status::TIMEOUT:          return "TIMEOUT";
        case Status::LINK_FAILURE:     return "LINK_FAILURE";
        default:                       return "UNKNOWN";
    }
}

class CannException : public std::runtime_error {
public:
    explicit CannException(const std::string& msg)
        : std::runtime_error(msg) {}
};

#define CANN_CHECK(expr) do {                                   \
    cann::Status _s = (expr);                                   \
    if (_s != cann::Status::SUCCESS) {                          \
        std::ostringstream _oss;                                \
        _oss << "CANN error: " << cann::StatusToString(_s)     \
             << " at " << __FILE__ << ":" << __LINE__;         \
        throw cann::CannException(_oss.str());                  \
    }                                                           \
} while (0)

#define CANN_VALIDATE_PARAM(cond) do {                          \
    if (!(cond)) {                                              \
        throw cann::CannException(                              \
            std::string("Invalid parameter: ") + #cond          \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                           \
} while (0)

} // namespace cann
