#ifndef AFFT_BIT_REV_PERM_PLAN_TYPE_HPP
#define AFFT_BIT_REV_PERM_PLAN_TYPE_HPP

namespace afft{
    enum class BitRevPermPlanType {
        n_indexes_equals_base_size_sqr,
        n_indexes_equals_2_base_size_sqr,
        n_indexes_equals_4_base_size_sqr,
        n_indexes_equals_8_base_size_sqr,
        indexes_mat_is_large_square,
        indexes_mat_is_large_nonsquare
    };
}

#endif