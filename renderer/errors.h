#pragma once

#include "commons.h"

#include <string>
#include <stdexcept>
#include <sstream>

// error handling, requires #include <stdexcept>
// copied from PyTorch

BEGIN_RENDERER_NAMESPACE
namespace detail
{
	inline std::string if_empty_then(std::string x, std::string y) {
		if (x.empty()) {
			return y;
		}
		else {
			return x;
		}
	}

	inline std::ostream& _str(std::ostream& ss) {
		return ss;
	}

	template <typename T>
	inline std::ostream& _str(std::ostream& ss, const T& t) {
		ss << t;
		return ss;
	}

	template <typename T, typename... Args>
	inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
		return _str(_str(ss, t), args...);
	}
}

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline std::string str(const Args&... args) {
	std::ostringstream ss;
	detail::_str(ss, args...);
	return ss.str();
}

// Specializations for already-a-string types.
template <>
inline std::string str(const std::string& str) {
	return str;
}
inline std::string str(const char* c_str) {
	return c_str;
}

END_RENDERER_NAMESPACE


#define CHECK_ERROR(cond, ...)                              \
  if (!(cond)) {                                            \
    throw std::runtime_error(                               \
      RENDERER_NAMESPACE ::detail::if_empty_then(           \
        RENDERER_NAMESPACE ::str(__VA_ARGS__),              \
        "Expected " #cond " to be true, but got false.  "   \
      )                                                     \
    );                                                      \
  }
