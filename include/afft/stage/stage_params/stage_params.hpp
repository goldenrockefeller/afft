#ifndef AFFT_STAGE_PARAMS_HPP
#define AFFT_STAGE_PARAMS_HPP

#include <cstddef>
#include "afft/stage/stage_params/ct_radix2_params.hpp"
#include "afft/stage/stage_params/ct_radix4_params.hpp"
#include "afft/stage/stage_params/s_radix_params.hpp"
#include "afft/stage/stage_params/conj_reverse_params.hpp"
#include "afft/stage/stage_params/deinterleave_params.hpp"
#include "afft/stage/stage_params/interleave_params.hpp"
#include "afft/stage/stage_params/apply_rfft_rotor_params.hpp"
#include "afft/stage/stage_params/complex_multiply_params.hpp"

namespace afft{
    template <typename Sample>
    union StageParams {
        CtRadix2Params<Sample> ct_r2;
        CtRadix4Params<Sample> ct_r4;
        SRadixParams<Sample> s_r;
        ConjReverseParams<Sample> conj_rev;
        DeinterleaveParams<Sample> deinterleave;
        InterleaveParams<Sample> interleave;
        ApplyRfftRotorParams<Sample> apply_rfft_rotor;
        ComplexMultiplyParams<Sample> complex_multiply;
    };      
}
#endif