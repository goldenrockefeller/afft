#ifndef AFFT_INTERLEAVE_HPP
#define AFFT_INTERLEAVE_HPP

#include <cstddef>

namespace afft {
    template <typename Sample>
    void interleave(Sample* out, const Sample* real, const Sample* imag, std::size_t id_start, std::size_t id_end) {
        for (std::size_t i = id_start; i < id_end; ++i) {
            out[2 * i] = real[i];
            out[2 * i + 1] = imag[i];
        }
    }
}

#endif
