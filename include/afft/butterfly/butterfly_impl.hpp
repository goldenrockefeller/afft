#ifndef AFFT_BUTTERFLY_IMPL_HPP
#define AFFT_BUTTERFLY_IMPL_HPP

#include <cstddef>
#include <vector>
#include <cstring>
#include "afft/butterfly/butterfly_plan.hpp"
#include "afft/radix/s_radix2.hpp"
#include "afft/radix/s_radix4.hpp"
#include "afft/radix/ct_radix2.hpp"
#include "afft/radix/ct_radix4.hpp"
#include "afft/radix/radix_stage.hpp"
#include "afft/radix/radix_type.hpp"
#include "afft/spec/bounded_spec.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class ButterflyImpl
    {
        using sample_spec = typename BoundedSpec<Spec, 0>::spec;
        using sample = typename Spec::sample; 

        ButterflyPlan<sample_spec, Allocator> plan_;
        mutable std::vector<sample, Allocator> buf_real_;
        mutable std::vector<sample, Allocator> buf_imag_;
        mutable std::vector<sample, Allocator> buf_real_2_;
        mutable std::vector<sample, Allocator> buf_imag_2_;

        public:
        explicit ButterflyImpl(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand), buf_real_(n_samples), buf_imag_(n_samples), buf_real_2_(n_samples), buf_imag_2_(n_samples) {}
        ButterflyImpl(std::size_t n_samples, std::size_t min_partition_len) 
            : plan_(n_samples, Spec::n_samples_per_operand, min_partition_len), 
            buf_real_(n_samples), 
            buf_imag_(n_samples), 
            buf_real_2_(n_samples), 
            buf_imag_2_(n_samples)  
        {}

        const ButterflyPlan<sample_spec, Allocator> &plan() const
        {
            return plan_;
        }

        template <bool Rescaling = false>
        static inline void eval(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag,
            sample *buf_real,
            sample *buf_imag,
            sample *buf_real_2,
            sample *buf_imag_2,
            const ButterflyPlan<sample_spec, Allocator> &plan)
        {

            using sample = sample;
            sample scaling_factor;

            sample *stage_real[3];
            sample *stage_imag[3];

            stage_real[0] = buf_real_2;
            stage_real[1] = buf_real;
            stage_real[2] = out_real;

            stage_imag[0] = buf_imag_2;
            stage_imag[1] = buf_imag;
            stage_imag[2] = out_imag;

            std::size_t s_in_id = 0;
            std::size_t s_out_id = 0;

            if (Rescaling){
                scaling_factor = plan.scaling_factor();
            }
            else {
                scaling_factor = sample(1);
            }

            if (plan.n_s_radix_stages() % 2 == 0)
            {
                s_out_id = 1;
            }
            else {
                s_out_id = 0;
            }

            if (plan.n_s_radix_stages() == 1 ) {
                if (
                    out_real != in_real 
                    && out_imag != in_real
                    && out_real != in_imag 
                    && out_imag != in_imag
                ) {
                    // Take stockholm output straight to output instead of buffer 
                    s_out_id = 2;
                }
            }

            // First stage, always Stockholm, takes input to buffer_1 / buffer_2, perform rescaling
            {
                const auto &radix_stage = plan.radix_stages()[0];

                switch (radix_stage.type)
                {
                case RadixType::ct_radix4:
                    break;
                case RadixType::s_radix4:
                    do_s_radix4_stage<Spec, Rescaling, false>(
                        stage_real[s_out_id],
                        stage_imag[s_out_id],
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.tw_real_c,
                        radix_stage.tw_imag_c,
                        radix_stage.tw_real_d,
                        radix_stage.tw_imag_d,
                        radix_stage.out_indexes,
                        radix_stage.in_indexes,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        radix_stage.log_subtwiddle_len,
                        radix_stage.n_samples,
                        scaling_factor);
                    break;
                case RadixType::ct_radix2:
                    break;
                case RadixType::s_radix2:
                    do_s_radix2_stage<Spec, Rescaling, false>(
                        stage_real[s_out_id],
                        stage_imag[s_out_id],
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.out_indexes,
                        radix_stage.in_indexes,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        radix_stage.log_subtwiddle_len,
                        radix_stage.n_samples,
                        scaling_factor);
                    break;
                }
            }

            auto s_out_real = out_real;
            auto s_out_imag = out_imag;
            
            if (plan.n_s_radix_stages() == 1 ) {
                if (plan.radix_stages().size() == 1 ) {
                    if (
                        out_real == in_real 
                        || out_imag == in_real
                        || out_real == in_imag 
                        || out_imag == in_imag
                    ) {
                        // Move stockholm from buffer to output
                        std::memcpy(out_real, buf_real_2, sizeof(sample) * plan.n_samples());
                        std::memcpy(out_imag, buf_imag_2, sizeof(sample) * plan.n_samples());
                    }
                }

                // Use cooley tukey to move data from buffer to output
                s_out_real = buf_real_2;
                s_out_imag = buf_imag_2;
            }

            // Stockholm flips between buffer and output, no rescaling, Cooley Tukey performs in-place on output
            for (std::size_t radix_stage_id = 1; radix_stage_id < plan.radix_stages().size(); radix_stage_id++)
            {
                
                const auto &radix_stage = plan.radix_stages()[radix_stage_id];

                auto ct_in_real = radix_stage.is_first_ct_radix_stage ? s_out_real : out_real;
                auto ct_in_imag = radix_stage.is_first_ct_radix_stage ? s_out_imag : out_imag;

                switch (radix_stage.type)
                {
                case RadixType::ct_radix4:
                    do_ct_radix4_stage<Spec>(
                        out_real,
                        out_imag,
                        ct_in_real,
                        ct_in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.tw_real_c,
                        radix_stage.tw_imag_c,
                        radix_stage.tw_real_d,
                        radix_stage.tw_imag_d,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        radix_stage.subtwiddle_len,
                        radix_stage.subtwiddle_start,
                        radix_stage.subtwiddle_end);
                    break;
                case RadixType::s_radix4:
                    s_in_id = s_out_id;
                    s_out_id = (s_out_id + 1) * (s_out_id < 2) + std::size_t(s_out_id == 2);
                    do_s_radix4_stage<Spec, false, true>(
                        stage_real[s_out_id],
                        stage_imag[s_out_id],
                        stage_real[s_in_id],
                        stage_imag[s_in_id],
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.tw_real_c,
                        radix_stage.tw_imag_c,
                        radix_stage.tw_real_d,
                        radix_stage.tw_imag_d,
                        radix_stage.out_indexes,
                        radix_stage.in_indexes,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        radix_stage.log_subtwiddle_len,
                        radix_stage.n_samples,
                        scaling_factor);
                    break;
                case RadixType::ct_radix2:
                    do_ct_radix2_stage<Spec>(
                        out_real,
                        out_imag,
                        ct_in_real,
                        ct_in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subtwiddle_len,
                        radix_stage.subtwiddle_start,
                        radix_stage.subtwiddle_end);
                    break;
                case RadixType::s_radix2:
                    s_in_id = s_out_id;
                    s_out_id = (s_out_id + 1) * (s_out_id < 2) + std::size_t(s_out_id == 2);
                    do_s_radix2_stage<Spec, false, true>(
                        stage_real[s_out_id],
                        stage_imag[s_out_id],
                        stage_real[s_in_id],
                        stage_imag[s_in_id],
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.out_indexes,
                        radix_stage.in_indexes,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        radix_stage.log_subtwiddle_len,
                        radix_stage.n_samples,
                        scaling_factor);
                    break;
                }
            }

        }

        template <bool Rescaling = false>
        void eval(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            eval<Rescaling>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                buf_real_.data(),
                buf_imag_.data(),
                buf_real_2_.data(),
                buf_imag_2_.data(),
                plan_);
        }
    };
}

#endif