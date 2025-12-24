#ifndef AFFT_S_RADIX4_HPP
#define AFFT_S_RADIX4_HPP

#include <cstddef>

#include "afft/radix/s_radix4_impl.hpp"
#include "afft/radix/radix_params/log_interleave_permute.hpp"

namespace afft
{
    template <typename Spec, bool Rescaling, bool HasTwiddles>
    inline void do_s_radix4_stage(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *twiddles,
        const std::size_t *out_indexes,
        const std::size_t *in_indexes,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        LogInterleavePermute log_interleave_permute,
        std::size_t n_samples,
        const typename Spec::sample &scaling_factor)
    {
        switch (log_interleave_permute)
        {
        case LogInterleavePermute::n0:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 0, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n1:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 1, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n2:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 2, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n3:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 3, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n4:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 4, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n5:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 5, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n6:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 6, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n7:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 7, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n8: // Up to interleave 256 support
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 8, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n9: // Up to interleave 256 support
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 9, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n10: // Up to interleave 256 support
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 10, false>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n0Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 0, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n1Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 1, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n2Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 2, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n3Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 3, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n4Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 4, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n5Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 5, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n6Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 6, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n7Permuting:
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 7, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n8Permuting: // Up to interleave 256 support
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 8, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n9Permuting: // Up to interleave 256 support
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 9, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case LogInterleavePermute::n10Permuting: // Up to interleave 256 support
            do_s_radix4_stage_impl<Spec, Rescaling, HasTwiddles, 10, true>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                out_indexes,
                in_indexes,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        }
    }
}

#endif