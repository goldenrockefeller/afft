#ifndef AFFT_LOG_INTERLEAVE_PERMUTE_HPP
#define AFFT_LOG_INTERLEAVE_PERMUTE_HPP

#include <cstddef>

namespace afft{
    enum class LogInterleavePermute{
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        n8,
        n9,
        n10,
        n0Permuting,
        n1Permuting,
        n2Permuting,
        n3Permuting,
        n4Permuting,
        n5Permuting,
        n6Permuting,
        n7Permuting,
        n8Permuting,
        n9Permuting,
        n10Permuting
    };

    LogInterleavePermute as_log_interleave_permute(std::size_t log_interleave_factor, bool permuting) {
        if (log_interleave_factor > 10) {
            throw std::out_of_range("log_interleave_factor must be between 0 and 10");
        }

        if (permuting) {
            return static_cast<LogInterleavePermute>(static_cast<int>(LogInterleavePermute::n0Permuting) + static_cast<int>(log_interleave_factor));
        } else {
            return static_cast<LogInterleavePermute>(static_cast<int>(LogInterleavePermute::n0) + static_cast<int>(log_interleave_factor));
        }
    }
}
#endif