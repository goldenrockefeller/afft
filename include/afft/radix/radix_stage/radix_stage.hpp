#ifndef AFFT_RADIX_STAGE_HPP
#define AFFT_RADIX_STAGE_HPP

#include <cstddef>
#include "afft/radix/radix_params/radix_params.hpp"
#include "afft/radix/radix_stage/radix_type.hpp"

namespace afft{
    template <typename Spec>
    struct RadixStage {
        RadixType type;
        RadixParams<Spec> params;
    };
}
#endif