#ifndef AFFT_COMPLEX_MULTIPLY_PARAMS_HPP
#define AFFT_COMPLEX_MULTIPLY_PARAMS_HPP

#include <cstddef>

namespace afft {
    template <typename Sample>
    struct ComplexMultiplyParams {
        std::size_t out_real_id;
        std::size_t out_imag_id;
        std::size_t in_real_a_id;
        std::size_t in_imag_a_id;
        std::size_t in_real_b_id;
        std::size_t in_imag_b_id;
        std::size_t id_start;
        std::size_t id_end;
        bool using_hermitian_packed_form;
    };
}

#endif
