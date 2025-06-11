#ifndef AFFT_S_RADIX2_HPP
#define AFFT_S_RADIX2_HPP

#include <cstddef>

#include "afft/radix/s_radix2_no_interleave.hpp"
#include "afft/radix/s_radix2_interleave.hpp"

namespace afft
{
    template <typename Spec, bool Rescaling, bool HasTwiddles>
    inline void do_s_radix2_stage(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *twiddles,
        const std::size_t *out_indexes,
        const std::size_t *in_indexes,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        std::size_t log_subtwiddle_len,
        std::size_t n_samples,
        const typename Spec::sample &scaling_factor)
    {
        if (out_indexes != in_indexes)
        {
            // This is the last Stockham Stage, do not inteleave!
            // Indexes are probably ordered bit reversed index for better cache optimality

            do_s_radix2_stage_no_interleave<Spec, Rescaling, HasTwiddles>(
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

            return;
        }

        switch (log_subtwiddle_len)
        {
        case 0:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 0>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 1:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 1>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 2:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 2>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 3:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 3>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 4:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 4>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 5:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 5>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 6:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 6>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 7:
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 7>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        case 8: // Up to interleave 256 support
            do_s_radix2_stage_interleave<Spec, Rescaling, HasTwiddles, 8>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                twiddles,
                subfft_id_start,
                subfft_id_end,
                n_samples,
                scaling_factor);
            break;
        }
    }
}

#endif