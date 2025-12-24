#ifndef AFFT_STAGE_PARAMS_HPP
#define AFFT_STAGE_PARAMS_HPP

#include <cstddef>
#include "afft/stage/stage_params/ct_radix2_params.hpp"
#include "afft/stage/stage_params/ct_radix4_params.hpp"
#include "afft/stage/stage_params/s_radix_params.hpp"

namespace afft{
    template <typename Sample>
    union StageParams {
        CtRadix2Params<Sample> ct_r2;
        CtRadix4Params<Sample> ct_r4;
        SRadixParams<Sample> s_r;
    };      
}
#endif