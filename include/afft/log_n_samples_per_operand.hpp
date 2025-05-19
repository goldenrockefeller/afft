#ifndef AFFT_LOG_N_SAMPLES_PER_OPERAND_HPP
#define AFFT_LOG_N_SAMPLES_PER_OPERAND_HPP

#include <stdexcept>  // for std::out_of_range
#include <cstddef>    // for std::size_t

namespace afft{
    enum class LogNSamplesPerOperand : unsigned char {
        n0 = 0,
        n1 = 1,
        n2 = 2,
        n3 = 3,
        n4 = 4,
        n5 = 5,
        n6 = 6,
        n7 = 7,
        n8 = 8  // Afft currently only support Operand Sizes up to 256 samples.
    };

    inline LogNSamplesPerOperand as_log_n_samples_per_operand(std::size_t value) {
        if (value > 8) {
            throw std::out_of_range("Invalid value for LogNSamplesPerOperand: must be 0–8");
        }
        return static_cast<LogNSamplesPerOperand>(value);
    }
}


#endif