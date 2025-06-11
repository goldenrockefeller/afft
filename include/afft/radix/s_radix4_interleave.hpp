#ifndef AFFT_S_RADIX4_INTERLEAVE_HPP
#define AFFT_S_RADIX4_INTERLEAVE_HPP

#include <cstddef>

namespace afft
{
    template <typename Spec, bool Rescaling, bool HasTwiddles, std::size_t LogInterleaveFactor>
    inline void do_s_radix4_stage_interleave(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *twiddles,
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

        operand alpha_real_c_op;
        operand alpha_real_d_op;
        operand alpha_imag_c_op;
        operand alpha_imag_d_op;
        operand beta_real_c_op;
        operand beta_real_d_op;
        operand beta_imag_c_op;
        operand beta_imag_d_op;

        operand scaling_factor_op(scaling_factor);

        operand tw_real_b_op;
        operand tw_imag_b_op;
        operand tw_real_c_op;
        operand tw_imag_c_op;
        operand tw_real_d_op;
        operand tw_imag_d_op;

        if (HasTwiddles)
        {
            Spec::load(tw_real_b_op, twiddles);
            Spec::load(tw_imag_b_op, twiddles + n_samples_per_operand);
            Spec::load(tw_real_c_op, twiddles + 2 * n_samples_per_operand);
            Spec::load(tw_imag_c_op, twiddles + 3 * n_samples_per_operand);
            Spec::load(tw_real_d_op, twiddles + 4 * n_samples_per_operand);
            Spec::load(tw_imag_d_op, twiddles + 5 * n_samples_per_operand);
        }

        auto box_size = Spec::n_samples_per_operand * 4;

        std::size_t in_a_offset = subfft_id_start * Spec::n_samples_per_operand;
        std::size_t in_b_offset = subfft_id_start * Spec::n_samples_per_operand + n_samples / 2;
        std::size_t in_c_offset = subfft_id_start * Spec::n_samples_per_operand + n_samples / 4;
        std::size_t in_d_offset = subfft_id_start * Spec::n_samples_per_operand + 3 * n_samples / 4;

        std::size_t out_a_offset = subfft_id_start * box_size;
        std::size_t out_b_offset = subfft_id_start * box_size + Spec::n_samples_per_operand;
        std::size_t out_c_offset = subfft_id_start * box_size + 2 * Spec::n_samples_per_operand;
        std::size_t out_d_offset = subfft_id_start * box_size + 3 * Spec::n_samples_per_operand;

        auto in_real_a = in_real + in_a_offset;
        auto in_imag_a = in_imag + in_a_offset;
        auto in_real_b = in_real + in_b_offset;
        auto in_imag_b = in_imag + in_b_offset;
        auto in_real_c = in_real + in_c_offset;
        auto in_imag_c = in_imag + in_c_offset;
        auto in_real_d = in_real + in_d_offset;
        auto in_imag_d = in_imag + in_d_offset;

        auto out_real_a = out_real + out_a_offset;
        auto out_imag_a = out_imag + out_a_offset;
        auto out_real_b = out_real + out_b_offset;
        auto out_imag_b = out_imag + out_b_offset;
        auto out_real_c = out_real + out_c_offset;
        auto out_imag_c = out_imag + out_c_offset;
        auto out_real_d = out_real + out_d_offset;
        auto out_imag_d = out_imag + out_d_offset;

        for (
            std::size_t subfft_id = subfft_id_start;
            subfft_id < subfft_id_end;
            subfft_id++)
        {
            
            // LOAD
            Spec::load(alpha_real_a_op, in_real_a);
            Spec::load(alpha_imag_a_op, in_imag_a);
            
            Spec::load(alpha_real_c_op, in_real_c);
            Spec::load(alpha_imag_c_op, in_imag_c);

            Spec::load(alpha_real_b_op, in_real_b);
            Spec::load(alpha_imag_b_op, in_imag_b);
            
            Spec::load(alpha_real_d_op, in_real_d);
            Spec::load(alpha_imag_d_op, in_imag_d);

            // PREFETCH
            Spec::prefetch(in_real_a + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_imag_a + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_real_b + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_imag_b + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_real_c + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_imag_c + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_real_d + Spec::prefetch_lookahead * Spec::n_samples_per_operand);
            Spec::prefetch(in_imag_d + Spec::prefetch_lookahead * Spec::n_samples_per_operand);

            Spec::prefetch(out_real_a + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_imag_a + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_real_b + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_imag_b + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_real_c + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_imag_c + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_real_d + Spec::prefetch_lookahead * box_size);
            Spec::prefetch(out_imag_d + Spec::prefetch_lookahead * box_size);

            // COMPUTE

            if (Rescaling) {
                alpha_real_b_op *= scaling_factor_op;
                alpha_imag_b_op *= scaling_factor_op;
                alpha_real_b_op *= scaling_factor_op;
                alpha_imag_b_op *= scaling_factor_op;
                alpha_real_c_op *= scaling_factor_op;
                alpha_imag_c_op *= scaling_factor_op;
                alpha_real_d_op *= scaling_factor_op;
                alpha_imag_d_op *= scaling_factor_op;
            }

            if (HasTwiddles)
            {
                beta_real_b_op = tw_imag_b_op * alpha_imag_b_op;
                beta_real_c_op = tw_imag_c_op * alpha_imag_c_op;
                beta_real_d_op = tw_imag_d_op * alpha_imag_d_op;

                beta_imag_b_op = tw_imag_b_op * alpha_real_b_op;
                beta_imag_c_op = tw_imag_c_op * alpha_real_c_op;
                beta_imag_d_op = tw_imag_d_op * alpha_real_d_op;

                alpha_real_b_op *= tw_real_b_op;
                alpha_imag_b_op *= tw_real_b_op;
                alpha_real_c_op *= tw_real_c_op;
                alpha_imag_c_op *= tw_real_c_op;
                alpha_real_d_op *= tw_real_d_op;
                alpha_imag_d_op *= tw_real_d_op;

                alpha_real_b_op -= beta_real_b_op;
                alpha_real_c_op -= beta_real_c_op;
                alpha_real_d_op -= beta_real_d_op;
                alpha_imag_b_op += beta_imag_b_op;
                alpha_imag_c_op += beta_imag_c_op;
                alpha_imag_d_op += beta_imag_d_op;
            }

            beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
            beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
            beta_real_c_op = alpha_real_c_op + alpha_real_d_op;
            beta_real_d_op = alpha_real_c_op - alpha_real_d_op;

            beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
            beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;
            beta_imag_c_op = alpha_imag_c_op + alpha_imag_d_op;
            beta_imag_d_op = alpha_imag_c_op - alpha_imag_d_op;

            alpha_real_a_op = beta_real_a_op + beta_real_c_op;
            alpha_real_b_op = beta_real_b_op + beta_imag_d_op;
            alpha_real_c_op = beta_real_a_op - beta_real_c_op;
            alpha_real_d_op = beta_real_b_op - beta_imag_d_op;

            alpha_imag_a_op = beta_imag_a_op + beta_imag_c_op;
            alpha_imag_b_op = beta_imag_b_op - beta_real_d_op;
            alpha_imag_c_op = beta_imag_a_op - beta_imag_c_op;
            alpha_imag_d_op = beta_imag_b_op + beta_real_d_op;

            Spec::template interleave4<LogInterleaveFactor>(
                beta_real_a_op, 
                beta_real_b_op, 
                beta_real_c_op,
                beta_real_d_op,
                alpha_real_a_op, 
                alpha_real_b_op, 
                alpha_real_c_op,
                alpha_real_d_op
            );

            Spec::template interleave4<LogInterleaveFactor>(
                beta_imag_a_op, 
                beta_imag_b_op, 
                beta_imag_c_op,
                beta_imag_d_op,
                alpha_imag_a_op, 
                alpha_imag_b_op, 
                alpha_imag_c_op,
                alpha_imag_d_op
            );

            // STORE

            Spec::store(out_real_a, beta_real_a_op);
            Spec::store(out_imag_a, beta_imag_a_op);
            Spec::store(out_real_b, beta_real_b_op);
            Spec::store(out_imag_b, beta_imag_b_op);
            Spec::store(out_real_c, beta_real_c_op);
            Spec::store(out_imag_c, beta_imag_c_op);
            Spec::store(out_real_d, beta_real_d_op);
            Spec::store(out_imag_d, beta_imag_d_op);
      
            // UPDATE OFFSET

            in_real_a += Spec::n_samples_per_operand;
            in_imag_a += Spec::n_samples_per_operand;
            in_real_b += Spec::n_samples_per_operand;
            in_imag_b += Spec::n_samples_per_operand;
            in_real_c += Spec::n_samples_per_operand;
            in_imag_c += Spec::n_samples_per_operand;
            in_real_d += Spec::n_samples_per_operand;
            in_imag_d += Spec::n_samples_per_operand;

            out_real_a += box_size;
            out_imag_a += box_size;
            out_real_b += box_size;
            out_imag_b += box_size;
            out_real_c += box_size;
            out_imag_c += box_size;
            out_real_d += box_size;
            out_imag_d += box_size;
        }
    }
}

#endif