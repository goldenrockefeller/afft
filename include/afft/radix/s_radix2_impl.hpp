#ifndef AFFT_S_RADIX2_IMPL_HPP
#define AFFT_S_RADIX2_IMPL_HPP

#include <cstddef>

namespace afft
{
    template <typename Spec, bool Rescaling, bool HasTwiddles, std::size_t LogInterleaveFactor>
    inline void do_s_radix2_stage_impl(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *twiddles,
        const std::size_t *out_indexes,
        const std::size_t *in_indexes,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        std::size_t n_samples,
        const typename Spec::sample &scaling_factor
    )
    {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;

        // DECLARE
        operand alpha_real_a_op;
        operand alpha_real_b_op;
        operand alpha_imag_a_op;
        operand alpha_imag_b_op;
        operand beta_real_a_op;
        operand beta_real_b_op;
        operand beta_imag_a_op;
        operand beta_imag_b_op;

        operand scaling_factor_op(scaling_factor);

        operand tw_real_b_op;
        operand tw_imag_b_op;

        if (HasTwiddles)
        {
            Spec::load(tw_real_b_op, twiddles);
            Spec::load(tw_imag_b_op, twiddles + n_samples_per_operand);
        }

        auto box_size = Spec::n_samples_per_operand * 2;

        auto n_samples_2 = n_samples / 2;

        for (
            std::size_t subfft_id = subfft_id_start;
            subfft_id < subfft_id_end;
            subfft_id++)
        {
            std::size_t out_index;
            std::size_t in_index;
            
            if ((1 << LogInterleaveFactor) < Spec::n_samples_per_operand) {
                out_index = subfft_id;
                in_index = subfft_id;
            } else {
                out_index = out_indexes[subfft_id];
                in_index = in_indexes[subfft_id];
            }

            std::size_t in_a_offset = in_index * Spec::n_samples_per_operand;
            std::size_t in_b_offset = in_index * Spec::n_samples_per_operand + n_samples_2;

            std::size_t out_a_offset = out_index * box_size;
            std::size_t out_b_offset = out_index * box_size + Spec::n_samples_per_operand;

            auto in_real_a = in_real + in_a_offset;
            auto in_imag_a = in_imag + in_a_offset;
            auto in_real_b = in_real + in_b_offset;
            auto in_imag_b = in_imag + in_b_offset;

            auto out_real_a = out_real + out_a_offset;
            auto out_imag_a = out_imag + out_a_offset;
            auto out_real_b = out_real + out_b_offset;
            auto out_imag_b = out_imag + out_b_offset;

            // LOAD
            Spec::load(alpha_real_a_op, in_real_a);
            Spec::load(alpha_imag_a_op, in_imag_a);
            Spec::load(alpha_real_b_op, in_real_b);
            Spec::load(alpha_imag_b_op, in_imag_b);

            // COMPUTE

            if (Rescaling) {
                alpha_real_b_op *= scaling_factor_op;
                alpha_imag_b_op *= scaling_factor_op;
                alpha_real_b_op *= scaling_factor_op;
                alpha_imag_b_op *= scaling_factor_op;
            }

            if (HasTwiddles)
            {
                beta_real_b_op = tw_imag_b_op * alpha_imag_b_op;
                beta_imag_b_op = tw_imag_b_op * alpha_real_b_op;

                alpha_real_b_op *= tw_real_b_op;
                alpha_imag_b_op *= tw_real_b_op;

                alpha_real_b_op -= beta_real_b_op;
                alpha_imag_b_op += beta_imag_b_op;
            
            }

            beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
            beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
            beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
            beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;

        
            if ((1 << LogInterleaveFactor) < Spec::n_samples_per_operand) {
                Spec::template interleave2<LogInterleaveFactor>(
                    alpha_real_a_op, 
                    alpha_real_b_op, 
                    beta_real_a_op, 
                    beta_real_b_op
                );

                Spec::template interleave2<LogInterleaveFactor>(
                    alpha_imag_a_op, 
                    alpha_imag_b_op, 
                    beta_imag_a_op, 
                    beta_imag_b_op
                );

                Spec::store(out_real_a, alpha_real_a_op);
                Spec::store(out_imag_a, alpha_imag_a_op);
                Spec::store(out_real_b, alpha_real_b_op);
                Spec::store(out_imag_b, alpha_imag_b_op);
            }
            else {
                Spec::store(out_real_a, beta_real_a_op);
                Spec::store(out_imag_a, beta_imag_a_op);
                Spec::store(out_real_b, beta_real_b_op);
                Spec::store(out_imag_b, beta_imag_b_op);
            }

            // PREFETCH
            if ((1 << LogInterleaveFactor) < Spec::n_samples_per_operand) {
                out_index = subfft_id + Spec::prefetch_lookahead;
                in_index = subfft_id + Spec::prefetch_lookahead;
            } else {
                out_index = out_indexes[subfft_id + Spec::prefetch_lookahead];
                in_index = in_indexes[subfft_id + Spec::prefetch_lookahead];

            in_a_offset = in_index * Spec::n_samples_per_operand;
            in_b_offset = in_index * Spec::n_samples_per_operand + n_samples_2;

            out_a_offset = out_index * box_size;
            out_b_offset = out_index * box_size + Spec::n_samples_per_operand;

            in_real_a = in_real + in_a_offset;
            in_imag_a = in_imag + in_a_offset;
            in_real_b = in_real + in_b_offset;
            in_imag_b = in_imag + in_b_offset;

            out_real_a = out_real + out_a_offset;
            out_imag_a = out_imag + out_a_offset;
            out_real_b = out_real + out_b_offset;
            out_imag_b = out_imag + out_b_offset;

            Spec::prefetch(in_real_a);
            Spec::prefetch(in_imag_a);
            Spec::prefetch(in_real_b);
            Spec::prefetch(in_imag_b);

            Spec::prefetch(out_real_a);
            Spec::prefetch(out_imag_a);
            Spec::prefetch(out_real_b);
            Spec::prefetch(out_imag_b);
            }

            if ((1 << LogInterleaveFactor) < Spec::n_samples_per_operand) { 
                // Do nothing
            } else {
                Spec::prefetch(out_indexes + subfft_id + 2 * Spec::prefetch_lookahead);
                Spec::prefetch(in_indexes + subfft_id +  2 * Spec::prefetch_lookahead);
            }
        }
    }
}

#endif