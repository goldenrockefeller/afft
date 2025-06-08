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
#include "afft/radix/radix_stage/radix_stage.hpp"
#include "afft/radix/radix_stage/radix_type.hpp"
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

        public:
        explicit ButterflyImpl(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand, Spec::prefetch_lookahead, Spec::min_partition_len), buf_real_(n_samples), buf_imag_(n_samples){}

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
            const ButterflyPlan<sample_spec, Allocator> &plan)
        {

            using sample = sample;
            sample scaling_factor;

            sample *data_real[3];
            sample *data_imag[3];
            
            if (plan.n_s_radix_stages() % 2 == 0){
                data_real[0] = nullptr;
                data_real[1] = buf_real;
                data_real[2] = out_real;

                data_imag[0] = nullptr;
                data_imag[1] = buf_imag;
                data_imag[2] = out_imag;
            }
            else {
                data_real[0] = nullptr;
                data_real[1] = out_real;
                data_real[2] = buf_real;

                data_imag[0] = nullptr;
                data_imag[1] = out_imag;
                data_imag[2] = buf_imag;
            }

            // First stage, always Stockholm, takes input to buffer_1 / buffer_2, perform rescaling, no twiddles
            {
                const auto &radix_stage = plan.radix_stages()[0];

                switch (radix_stage.type)
                {
                case RadixType::ct_radix4:
                    break;
                case RadixType::s_radix4:
                    {
                        auto &params = radix_stage.params.s_r4;
                        do_s_radix4_stage<Spec, Rescaling, false>(
                            data_real[params.output_id],
                            data_imag[params.output_id],
                            in_real,
                            in_imag,
                            params.tw_real_b_0,
                            params.tw_imag_b_0,
                            params.tw_real_c_0,
                            params.tw_imag_c_0,
                            params.tw_real_d_0,
                            params.tw_imag_d_0,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_subtwiddle_len,
                            plan.n_samples(),
                            plan.scaling_factor());
                    }
                    break;
                case RadixType::ct_radix2:
                    break;
                case RadixType::s_radix2:
                    {
                        
                        auto &params = radix_stage.params.s_r2;
                        do_s_radix2_stage<Spec, Rescaling, false>(
                            data_real[params.output_id],
                            data_imag[params.output_id],
                            in_real,
                            in_imag,
                            params.tw_real_b_0,
                            params.tw_imag_b_0,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_subtwiddle_len,
                            plan.n_samples(),
                            plan.scaling_factor());
                    }
                    break;
                }                
            }

            // Stockholm flips between buffer and output, no rescaling, Cooley Tukey performs in-place on output
            for (std::size_t radix_data_id = 1; radix_data_id < plan.radix_stages().size(); radix_data_id++)
            {

                const auto &radix_stage = plan.radix_stages()[radix_data_id];
                switch (radix_stage.type)
                {
                case RadixType::ct_radix4:
                    {
                        auto &params = radix_stage.params.ct_r4;
                        do_ct_radix4_stage<Spec>(
                            out_real,
                            out_imag,
                            out_real,
                            out_imag,
                            params.tw_real_b_0,
                            params.tw_imag_b_0,
                            params.tw_real_c_0,
                            params.tw_imag_c_0,
                            params.tw_real_d_0,
                            params.tw_imag_d_0,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.subtwiddle_len,
                            params.subtwiddle_start,
                            params.subtwiddle_end,
                            params.stride);
                    }
                    break;
                case RadixType::s_radix4:
                    {
                        auto &params = radix_stage.params.s_r4;
                        do_s_radix4_stage<Spec, false, true>(
                            data_real[params.output_id],
                            data_imag[params.output_id],
                            data_real[params.input_id],
                            data_imag[params.input_id],
                            params.tw_real_b_0,
                            params.tw_imag_b_0,
                            params.tw_real_c_0,
                            params.tw_imag_c_0,
                            params.tw_real_d_0,
                            params.tw_imag_d_0,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_subtwiddle_len,
                            plan.n_samples(),
                            plan.scaling_factor());
                    }
                    break;
                case RadixType::ct_radix2:
                    {
                        auto &params = radix_stage.params.ct_r2;
                        do_ct_radix2_stage<Spec>(
                            out_real,
                            out_imag,
                            out_real,
                            out_imag,
                            params.tw_real_b_0,
                            params.tw_imag_b_0,
                            params.subtwiddle_len,
                            params.subtwiddle_start,
                            params.subtwiddle_end,
                            params.stride);
                    }
                    break;
                case RadixType::s_radix2:
                    {
                        auto &params = radix_stage.params.s_r2;
                        do_s_radix2_stage<Spec, false, true>(
                            data_real[params.output_id],
                            data_imag[params.output_id],
                            data_real[params.input_id],
                            data_imag[params.input_id],
                            params.tw_real_b_0,
                            params.tw_imag_b_0,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_subtwiddle_len,
                            plan.n_samples(),
                            plan.scaling_factor());
                    }
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
                plan_);
        }
    };
}

#endif