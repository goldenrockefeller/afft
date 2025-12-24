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
        s_radix2_init_rescale
    };
}

#endif