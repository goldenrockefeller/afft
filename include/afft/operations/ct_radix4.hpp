#ifndef AFFT_CT_RADIX4_HPP
#define AFFT_CT_RADIX4_HPP

#include <cstddef>

namespace afft{
    template<typename Spec>
    inline void do_ct_radix4_stage(
        typename Spec::sample* out_real, 
        typename Spec::sample* out_imag, 
        const typename Spec::sample* twiddles,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        std::size_t subtwiddle_len,
        std::size_t subtwiddle_start,
        std::size_t subtwiddle_end,
        std::ptrdiff_t output_offset = 0
    ) {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
        const std::size_t subfft_len = 4 * subtwiddle_len;
        const std::size_t data_stride = n_samples_per_operand ;
        const std::size_t twiddles_stride = 6 * n_samples_per_operand ;

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
        
        operand tw_real_b_op;
        operand tw_imag_b_op;
        operand tw_real_c_op;
        operand tw_imag_c_op;
        operand tw_real_d_op;
        operand tw_imag_d_op;

        for (
            std::size_t subfft_id = subfft_id_start;
            subfft_id < subfft_id_end;
            subfft_id++
        ) {
            std::size_t a_offset = subfft_id * subfft_len + subtwiddle_start;
            std::size_t b_offset = a_offset + subtwiddle_len;
            std::size_t c_offset = a_offset + 2 * subtwiddle_len;
            std::size_t d_offset = a_offset + 3 * subtwiddle_len;

            // OFFSET
            auto out_real_a = out_real + a_offset;
            auto out_imag_a = out_imag + a_offset;
            auto out_real_b = out_real + b_offset;
            auto out_imag_b = out_imag + b_offset;
            auto out_real_c = out_real + c_offset;
            auto out_imag_c = out_imag + c_offset;
            auto out_real_d = out_real + d_offset;
            auto out_imag_d = out_imag + d_offset;

            auto tw_real_b = twiddles + 6 * subtwiddle_start;
            auto tw_imag_b = tw_real_b + n_samples_per_operand;
            auto tw_real_c = tw_real_b + 2 * n_samples_per_operand;
            auto tw_imag_c = tw_real_b + 3 * n_samples_per_operand;
            auto tw_real_d = tw_real_b + 4 * n_samples_per_operand;
            auto tw_imag_d = tw_real_b + 5 * n_samples_per_operand;

            for (
                std::size_t i = subtwiddle_start; 
                i < subtwiddle_end;
                i += data_stride
            ) {                
                //LOAD
                Spec::load(alpha_real_a_op, out_real_a);
                Spec::load(alpha_imag_a_op, out_imag_a);
                Spec::load(alpha_real_b_op, out_real_b);
                Spec::load(alpha_imag_b_op, out_imag_b);
                Spec::load(alpha_real_c_op, out_real_c);
                Spec::load(alpha_imag_c_op, out_imag_c);
                Spec::load(alpha_real_d_op, out_real_d);
                Spec::load(alpha_imag_d_op, out_imag_d);

                Spec::load(tw_real_b_op, tw_real_b);
                Spec::load(tw_imag_b_op, tw_imag_b);
                Spec::load(tw_real_c_op, tw_real_c);
                Spec::load(tw_imag_c_op, tw_imag_c);
                Spec::load(tw_real_d_op, tw_real_d);
                Spec::load(tw_imag_d_op, tw_imag_d);

                // COMPUTE 
                
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

                // STORE

                Spec::store(out_real_a, alpha_real_a_op);
                Spec::store(out_imag_a, alpha_imag_a_op);
                Spec::store(out_real_b, alpha_real_b_op);
                Spec::store(out_imag_b, alpha_imag_b_op);
                Spec::store(out_real_c, alpha_real_c_op);
                Spec::store(out_imag_c, alpha_imag_c_op);
                Spec::store(out_real_d, alpha_real_d_op);
                Spec::store(out_imag_d, alpha_imag_d_op);

                // UPDATE OFFSET
                out_real_a += data_stride;
                out_imag_a += data_stride;
                out_real_b += data_stride;
                out_imag_b += data_stride;
                out_real_c += data_stride;
                out_imag_c += data_stride;
                out_real_d += data_stride;
                out_imag_d += data_stride;

                tw_real_b += twiddles_stride;
                tw_imag_b += twiddles_stride;
                tw_real_c += twiddles_stride;
                tw_imag_c += twiddles_stride;
                tw_real_d += twiddles_stride;
                tw_imag_d += twiddles_stride;
            }
        }
    }
}

#endif