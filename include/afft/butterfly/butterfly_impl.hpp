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

        public:

        template <bool Rescaling = false>
        static inline void eval(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag,
            const ButterflyPlan<sample_spec, Allocator> &plan,
            sample *buf)
        {

            using sample = sample;
            sample scaling_factor;

            if (plan.n_samples() <= 1) {
                *out_real = *in_real;
                *out_imag = *in_imag;
            }

            sample *s_io_real[2];
            sample *s_io_imag[2];

            s_io_real[0] = buf;
            s_io_real[1] = out_real;

            s_io_imag[0] = buf + plan.n_samples();
            s_io_imag[1] = out_imag;

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
                            s_io_real[params.output_id],
                            s_io_imag[params.output_id],
                            in_real,
                            in_imag,
                            params.twiddles,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
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
                            s_io_real[params.output_id],
                            s_io_imag[params.output_id],
                            in_real,
                            in_imag,
                            params.twiddles,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
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
                            params.twiddles,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.subtwiddle_len,
                            params.subtwiddle_start,
                            params.subtwiddle_end);
                    }
                    break;
                case RadixType::s_radix4:
                    {
                        auto &params = radix_stage.params.s_r4;
                        do_s_radix4_stage<Spec, false, true>(
                            s_io_real[params.output_id],
                            s_io_imag[params.output_id],
                            s_io_real[params.input_id],
                            s_io_imag[params.input_id],
                            params.twiddles,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
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
                            params.twiddles,
                            params.subtwiddle_len,
                            params.subtwiddle_start,
                            params.subtwiddle_end);
                    }
                    break;
                case RadixType::s_radix2:
                    {
                        auto &params = radix_stage.params.s_r2;
                        do_s_radix2_stage<Spec, false, true>(
                            s_io_real[params.output_id],
                            s_io_imag[params.output_id],
                            s_io_real[params.input_id],
                            s_io_imag[params.input_id],
                            params.twiddles,
                            params.out_indexes,
                            params.in_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            plan.n_samples(),
                            plan.scaling_factor());
                    }
                    break;
                }
            }
        }
    };
}

#endif