#ifndef AFFT_STAGE_TYPE_HPP
#define AFFT_STAGE_TYPE_HPP

namespace afft{
    enum class StageType {
        ct_radix4,
        ct_radix2,
        s_radix4,
        s_radix4_init,
        s_radix4_init_rescale,
        s_radix2,
        s_radix2_init,
        s_radix2_init_rescale,
        conj_reverse,
        deinterleave,
        interleave,
        apply_inv_rfft_rotor,
        apply_rfft_rotor,
        complex_multiply
    };
}

#endif