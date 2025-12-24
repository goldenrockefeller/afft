#ifndef AFFT_STAGE_HPP
#define AFFT_STAGE_HPP

#include <cstddef>
#include "afft/stage/stage_params/stage_params.hpp"
#include "afft/stage/stage_type.hpp"

namespace afft{
    template <typename Sample>
    struct Stage {
        StageType type;
        StageParams<Sample> params;
    };
}
#endif