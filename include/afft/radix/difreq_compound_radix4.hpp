#ifndef AFFT_DIFREQ_COMPOUND_RADIX4_HPP
#define AFFT_DIFREQ_COMPOUND_RADIX4_HPP

#include <cstddef>
#include "afft/radix/compound_radix_twiddle_loader.hpp"

namespace afft
{

    template <typename Spec, std::size_t StageFactor, std::size_t StageId>
    struct ComputeDifreqCompoundRadix4Substage
    {
        static inline void eval(
            typename Spec::operand &alpha_real_a_op,
            typename Spec::operand &alpha_real_b_op,
            typename Spec::operand &alpha_real_c_op,
            typename Spec::operand &alpha_real_d_op,
            typename Spec::operand &alpha_imag_a_op,
            typename Spec::operand &alpha_imag_b_op,
            typename Spec::operand &alpha_imag_c_op,
            typename Spec::operand &alpha_imag_d_op,
            typename Spec::operand &beta_real_a_op,
            typename Spec::operand &beta_real_b_op,
            typename Spec::operand &beta_real_c_op,
            typename Spec::operand &beta_real_d_op,
            typename Spec::operand &beta_imag_a_op,
            typename Spec::operand &beta_imag_b_op,
            typename Spec::operand &beta_imag_c_op,
            typename Spec::operand &beta_imag_d_op,
            const typename Spec::operand *tw_real_b_op,
            const typename Spec::operand *tw_imag_b_op,
            const typename Spec::operand *tw_real_c_op,
            const typename Spec::operand *tw_imag_c_op,
            const typename Spec::operand *tw_real_d_op,
            const typename Spec::operand *tw_imag_d_op)
        {
            ComputeDifreqCompoundRadix4Substage<Spec, StageFactor / 4, StageId + 1>::eval(
                alpha_real_a_op,
                alpha_real_b_op,
                alpha_real_c_op,
                alpha_real_d_op,
                alpha_imag_a_op,
                alpha_imag_b_op,
                alpha_imag_c_op,
                alpha_imag_d_op,
                beta_real_a_op,
                beta_real_b_op,
                beta_real_c_op,
                beta_real_d_op,
                beta_imag_a_op,
                beta_imag_b_op,
                beta_imag_c_op,
                beta_imag_d_op,
                tw_real_b_op,
                tw_imag_b_op,
                tw_real_c_op,
                tw_imag_c_op,
                tw_real_d_op,
                tw_imag_d_op);

            beta_real_a_op = alpha_real_a_op + alpha_real_c_op;
            beta_real_b_op = alpha_real_b_op + alpha_real_d_op;
            beta_real_c_op = alpha_real_a_op - alpha_real_c_op;
            beta_real_d_op = alpha_imag_b_op - alpha_imag_d_op;

            beta_imag_a_op = alpha_imag_a_op + alpha_imag_c_op;
            beta_imag_b_op = alpha_imag_b_op + alpha_imag_d_op;
            beta_imag_c_op = alpha_imag_a_op - alpha_imag_c_op;
            beta_imag_d_op = alpha_real_d_op - alpha_real_b_op;

            alpha_real_a_op = beta_real_a_op + beta_real_b_op;
            alpha_real_b_op = beta_real_a_op - beta_real_b_op;
            alpha_real_c_op = beta_real_c_op + beta_real_d_op;
            alpha_real_d_op = beta_real_c_op - beta_real_d_op;

            alpha_imag_a_op = beta_imag_a_op + beta_imag_b_op;
            alpha_imag_b_op = beta_imag_a_op - beta_imag_b_op;
            alpha_imag_c_op = beta_imag_c_op + beta_imag_d_op;
            alpha_imag_d_op = beta_imag_c_op - beta_imag_d_op;

            beta_real_b_op = tw_imag_b_op[StageId] * alpha_imag_b_op;
            beta_imag_b_op = tw_imag_b_op[StageId] * alpha_real_b_op;
            beta_real_c_op = tw_imag_c_op[StageId] * alpha_imag_c_op;
            beta_imag_c_op = tw_imag_c_op[StageId] * alpha_real_c_op;
            beta_real_d_op = tw_imag_d_op[StageId] * alpha_imag_d_op;
            beta_imag_d_op = tw_imag_d_op[StageId] * alpha_real_d_op;

            alpha_real_b_op *= tw_real_b_op[StageId];
            alpha_imag_b_op *= tw_real_b_op[StageId];
            alpha_real_c_op *= tw_real_c_op[StageId];
            alpha_imag_c_op *= tw_real_c_op[StageId];
            alpha_real_d_op *= tw_real_d_op[StageId];
            alpha_imag_d_op *= tw_real_d_op[StageId];

            beta_real_b_op = alpha_real_b_op - beta_real_b_op;
            beta_imag_b_op += alpha_imag_b_op;
            beta_real_c_op = alpha_real_c_op - beta_real_c_op;
            beta_imag_c_op += alpha_imag_c_op;
            beta_real_d_op = alpha_real_d_op - beta_real_d_op;
            beta_imag_d_op += alpha_imag_d_op;

            beta_real_a_op = alpha_real_a_op;
            beta_imag_a_op = alpha_imag_a_op;

            // Spec::interleave4(
            //     alpha_real_a_op, alpha_imag_a_op,
            //     alpha_real_b_op, alpha_imag_b_op,
            //     alpha_real_c_op, alpha_imag_c_op,
            //     alpha_real_d_op, alpha_imag_d_op,
            //     beta_real_a_op, beta_imag_a_op,
            //     beta_real_b_op, beta_imag_b_op,
            //     beta_real_c_op, beta_imag_c_op,
            //     beta_real_d_op, beta_imag_d_op);
        }
    };

    template <typename Spec, std::size_t StageId>
    struct ComputeDifreqCompoundRadix4Substage<Spec, 1, StageId>
    {
        static inline void eval(
            typename Spec::operand &alpha_real_a_op,
            typename Spec::operand &alpha_real_b_op,
            typename Spec::operand &alpha_real_c_op,
            typename Spec::operand &alpha_real_d_op,
            typename Spec::operand &alpha_imag_a_op,
            typename Spec::operand &alpha_imag_b_op,
            typename Spec::operand &alpha_imag_c_op,
            typename Spec::operand &alpha_imag_d_op,
            typename Spec::operand &beta_real_a_op,
            typename Spec::operand &beta_real_b_op,
            typename Spec::operand &beta_real_c_op,
            typename Spec::operand &beta_real_d_op,
            typename Spec::operand &beta_imag_a_op,
            typename Spec::operand &beta_imag_b_op,
            typename Spec::operand &beta_imag_c_op,
            typename Spec::operand &beta_imag_d_op,
            const typename Spec::operand *tw_real_b_op,
            const typename Spec::operand *tw_imag_b_op,
            const typename Spec::operand *tw_real_c_op,
            const typename Spec::operand *tw_imag_c_op,
            const typename Spec::operand *tw_real_d_op,
            const typename Spec::operand *tw_imag_d_op)
        {
            // Do nothing
        }
    };

    template <typename Spec, std::size_t StageId>
    struct ComputeDifreqCompoundRadix4Substage<Spec, 2, StageId>
    {
        static inline void eval(
            typename Spec::operand &alpha_real_a_op,
            typename Spec::operand &alpha_real_b_op,
            typename Spec::operand &alpha_real_c_op,
            typename Spec::operand &alpha_real_d_op,
            typename Spec::operand &alpha_imag_a_op,
            typename Spec::operand &alpha_imag_b_op,
            typename Spec::operand &alpha_imag_c_op,
            typename Spec::operand &alpha_imag_d_op,
            typename Spec::operand &beta_real_a_op,
            typename Spec::operand &beta_real_b_op,
            typename Spec::operand &beta_real_c_op,
            typename Spec::operand &beta_real_d_op,
            typename Spec::operand &beta_imag_a_op,
            typename Spec::operand &beta_imag_b_op,
            typename Spec::operand &beta_imag_c_op,
            typename Spec::operand &beta_imag_d_op,
            const typename Spec::operand *tw_real_b_op,
            const typename Spec::operand *tw_imag_b_op,
            const typename Spec::operand *tw_real_c_op,
            const typename Spec::operand *tw_imag_c_op,
            const typename Spec::operand *tw_real_d_op,
            const typename Spec::operand *tw_imag_d_op)
        {

            beta_real_a_op = alpha_real_a_op + alpha_real_c_op;
            beta_real_c_op = alpha_real_a_op - alpha_real_c_op;
            beta_imag_a_op = alpha_imag_a_op + alpha_imag_c_op;
            beta_imag_c_op = alpha_imag_a_op - alpha_imag_c_op;
            beta_real_b_op = alpha_real_b_op + alpha_real_d_op;
            beta_real_d_op = alpha_real_b_op - alpha_real_d_op;
            beta_imag_b_op = alpha_imag_b_op + alpha_imag_d_op;
            beta_imag_d_op = alpha_imag_b_op - alpha_imag_d_op;

            alpha_real_c_op = tw_imag_c_op[StageId] * beta_imag_c_op;
            alpha_imag_c_op = tw_imag_c_op[StageId] * beta_real_c_op;
            alpha_real_d_op = tw_imag_d_op[StageId] * beta_imag_d_op;
            alpha_imag_d_op = tw_imag_d_op[StageId] * beta_real_d_op;

            beta_real_c_op *= tw_real_c_op[StageId];
            beta_imag_c_op *= tw_real_c_op[StageId];
            beta_real_d_op *= tw_real_d_op[StageId];
            beta_imag_d_op *= tw_real_d_op[StageId];

            beta_real_c_op -= alpha_real_c_op;
            beta_imag_c_op += alpha_imag_c_op;
            beta_real_d_op -= alpha_real_d_op;
            beta_imag_d_op += alpha_imag_d_op;

            // Spec::interleave2(
            //     alpha_real_a_op, alpha_imag_a_op,
            //     alpha_real_b_op, alpha_imag_b_op,
            //     beta_real_a_op, beta_imag_a_op,
            //     beta_real_c_op, beta_imag_c_op);

            // Spec::interleave2(
            //     alpha_real_c_op, alpha_imag_c_op,
            //     alpha_real_d_op, alpha_imag_d_op,
            //     beta_real_b_op, beta_imag_b_op,
            //     beta_real_d_op, beta_imag_d_op);
        }
    };

    template <typename Spec, bool Rescaling>
    inline void do_difreq_compound_radix4_stage(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *tw_real,
        const typename Spec::sample *tw_imag,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        const typename Spec::sample &scaling_factor)
    {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
        const std::size_t subfft_len = 4 * n_samples_per_operand;

        // DECLARE
        operand alpha_real_a_op;
        operand alpha_real_b_op;
        operand alpha_real_c_op;
        operand alpha_real_d_op;

        operand alpha_imag_a_op;
        operand alpha_imag_b_op;
        operand alpha_imag_c_op;
        operand alpha_imag_d_op;

        operand beta_real_a_op;
        operand beta_real_b_op;
        operand beta_real_c_op;
        operand beta_real_d_op;

        operand beta_imag_a_op;
        operand beta_imag_b_op;
        operand beta_imag_c_op;
        operand beta_imag_d_op;

        // 5 twiddle stages should be enough for an operand size of 256 samples!
        operand tw_real_b_op[5];
        operand tw_imag_b_op[5];
        operand tw_real_c_op[5];
        operand tw_imag_c_op[5];
        operand tw_real_d_op[5];
        operand tw_imag_d_op[5];

        CompoundRadixTwiddleLoader<Spec, n_samples_per_operand, 0>::load_compound_radix_tw4(
            tw_real_b_op,
            tw_imag_b_op,
            tw_real_c_op,
            tw_imag_c_op,
            tw_real_d_op,
            tw_imag_d_op,
            tw_real,
            tw_imag);

        std::size_t a_offset = subfft_id_start * subfft_len;
        std::size_t b_offset = a_offset + n_samples_per_operand;
        std::size_t c_offset = a_offset + 2 * n_samples_per_operand;
        std::size_t d_offset = a_offset + 3 * n_samples_per_operand;

        // OFFSET
        auto in_real_a = in_real + a_offset;
        auto in_imag_a = in_imag + a_offset;
        auto in_real_b = in_real + b_offset;
        auto in_imag_b = in_imag + b_offset;
        auto in_real_c = in_real + c_offset;
        auto in_imag_c = in_imag + c_offset;
        auto in_real_d = in_real + d_offset;
        auto in_imag_d = in_imag + d_offset;

        auto out_real_a = out_real + a_offset;
        auto out_imag_a = out_imag + a_offset;
        auto out_real_b = out_real + b_offset;
        auto out_imag_b = out_imag + b_offset;
        auto out_real_c = out_real + c_offset;
        auto out_imag_c = out_imag + c_offset;
        auto out_real_d = out_real + d_offset;
        auto out_imag_d = out_imag + d_offset;

        for (
            std::size_t subfft_id = subfft_id_start;
            subfft_id < subfft_id_end;
            subfft_id++)
        {

            // LOAD
            Spec::load(alpha_real_a_op, in_real_a);
            Spec::load(alpha_imag_a_op, in_imag_a);
            Spec::load(alpha_real_b_op, in_real_b);
            Spec::load(alpha_imag_b_op, in_imag_b);
            Spec::load(alpha_real_c_op, in_real_c);
            Spec::load(alpha_imag_c_op, in_imag_c);
            Spec::load(alpha_real_d_op, in_real_d);
            Spec::load(alpha_imag_d_op, in_imag_d);

            // COMPUTE

            ComputeDifreqCompoundRadix4Substage<Spec, n_samples_per_operand, 0>::eval(
                alpha_real_a_op,
                alpha_real_b_op,
                alpha_real_c_op,
                alpha_real_d_op,
                alpha_imag_a_op,
                alpha_imag_b_op,
                alpha_imag_c_op,
                alpha_imag_d_op,
                beta_real_a_op,
                beta_real_b_op,
                beta_real_c_op,
                beta_real_d_op,
                beta_imag_a_op,
                beta_imag_b_op,
                beta_imag_c_op,
                beta_imag_d_op,
                tw_real_b_op,
                tw_imag_b_op,
                tw_real_c_op,
                tw_imag_c_op,
                tw_real_d_op,
                tw_imag_d_op);

            beta_real_a_op = alpha_real_a_op + alpha_real_c_op;
            beta_real_b_op = alpha_real_b_op + alpha_real_d_op;
            beta_real_c_op = alpha_real_a_op - alpha_real_c_op;
            beta_real_d_op = alpha_imag_b_op - alpha_imag_d_op;

            beta_imag_a_op = alpha_imag_a_op + alpha_imag_c_op;
            beta_imag_b_op = alpha_imag_b_op + alpha_imag_d_op;
            beta_imag_c_op = alpha_imag_a_op - alpha_imag_c_op;
            beta_imag_d_op = alpha_real_d_op - alpha_real_b_op;

            alpha_real_a_op = beta_real_a_op + beta_real_b_op;
            alpha_real_b_op = beta_real_a_op - beta_real_b_op;
            alpha_real_c_op = beta_real_c_op + beta_real_d_op;
            alpha_real_d_op = beta_real_c_op - beta_real_d_op;

            alpha_imag_a_op = beta_imag_a_op + beta_imag_b_op;
            alpha_imag_b_op = beta_imag_a_op - beta_imag_b_op;
            alpha_imag_c_op = beta_imag_c_op + beta_imag_d_op;
            alpha_imag_d_op = beta_imag_c_op - beta_imag_d_op;

            Spec::store(out_real_a, alpha_real_a_op);
            Spec::store(out_imag_a, alpha_imag_a_op);
            Spec::store(out_real_b, alpha_real_b_op);
            Spec::store(out_imag_b, alpha_imag_b_op);
            Spec::store(out_real_c, alpha_real_c_op);
            Spec::store(out_imag_c, alpha_imag_c_op);
            Spec::store(out_real_d, alpha_real_d_op);
            Spec::store(out_imag_d, alpha_imag_d_op);

            if (Rescaling)
            {
                alpha_real_a_op *= scaling_factor;
                alpha_imag_a_op *= scaling_factor;
                alpha_real_b_op *= scaling_factor;
                alpha_imag_b_op *= scaling_factor;
                alpha_real_c_op *= scaling_factor;
                alpha_imag_c_op *= scaling_factor;
                alpha_real_d_op *= scaling_factor;
                alpha_imag_d_op *= scaling_factor;
            }

            if (n_samples_per_operand == 1)
            {
                // STORE
                Spec::store(out_real_a, alpha_real_a_op);
                Spec::store(out_imag_a, alpha_imag_a_op);
                Spec::store(out_real_b, alpha_real_b_op);
                Spec::store(out_imag_b, alpha_imag_b_op);
                Spec::store(out_real_c, alpha_real_c_op);
                Spec::store(out_imag_c, alpha_imag_c_op);
                Spec::store(out_real_d, alpha_real_d_op);
                Spec::store(out_imag_d, alpha_imag_d_op);
            }
            else
            {
                // Spec::interleave4(
                //     beta_real_a_op, beta_imag_a_op,
                //     beta_real_b_op, beta_imag_b_op,
                //     beta_real_c_op, beta_imag_c_op,
                //     beta_real_d_op, beta_imag_d_op,
                //     alpha_real_a_op, alpha_imag_a_op,
                //     alpha_real_b_op, alpha_imag_b_op,
                //     alpha_real_c_op, alpha_imag_c_op,
                //     alpha_real_d_op, alpha_imag_d_op);

                // STORE
                Spec::store(out_real_a, beta_real_a_op);
                Spec::store(out_imag_a, beta_imag_a_op);
                Spec::store(out_real_b, beta_real_b_op);
                Spec::store(out_imag_b, beta_imag_b_op);
                Spec::store(out_real_c, beta_real_c_op);
                Spec::store(out_imag_c, beta_imag_c_op);
                Spec::store(out_real_d, beta_real_d_op);
                Spec::store(out_imag_d, beta_imag_d_op);
            }

            // UPDATE OFFSET
            in_real_a += subfft_len;
            in_imag_a += subfft_len;
            in_real_b += subfft_len;
            in_imag_b += subfft_len;
            in_real_c += subfft_len;
            in_imag_c += subfft_len;
            in_real_d += subfft_len;
            in_imag_d += subfft_len;

            out_real_a += subfft_len;
            out_imag_a += subfft_len;
            out_real_b += subfft_len;
            out_imag_b += subfft_len;
            out_real_c += subfft_len;
            out_imag_c += subfft_len;
            out_real_d += subfft_len;
            out_imag_d += subfft_len;
        }
    }
}

#endif