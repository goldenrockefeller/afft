#ifndef AFFT_RADIX_PARAMS_HPP
#define AFFT_RADIX_PARAMS_HPP

#include <cstddef>
#include "afft/radix/radix_params/ct_radix2_params.hpp"
#include "afft/radix/radix_params/ct_radix4_params.hpp"
#include "afft/radix/radix_params/s_radix2_params.hpp"
#include "afft/radix/radix_params/s_radix4_params.hpp"

namespace afft{
    template <typename Spec>
    union RadixParams {
        CtRadix2Params<Spec> ct_r2;
        CtRadix4Params<Spec> ct_r4;
        SRadix2Params<Spec> s_r2;
        SRadix4Params<Spec> s_r4;
    };      
}
#endif