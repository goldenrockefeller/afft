
#ifndef AFFT_DEINTERLEAVE_HPP
#define AFFT_DEINTERLEAVE_HPP

#include <cstddef>

namespace afft {
    template <typename Sample>
    void deinterleave(Sample* real, Sample* imag, const Sample* in, std::size_t id_start, std::size_t id_end) {
        for (std::size_t i = id_start; i < id_end; ++i) {
            real[i] = in[2 * i];
            imag[i] = in[2 * i + 1];
        }
    }
}

#endif
