#ifndef AFFT_BUTTERFLY_IMPL_HPP
#define AFFT_BUTTERFLY_IMPL_HPP

#include <cstddef>
#include <vector>
#include "afft/butterfly/butterfly_plan.hpp"
#include "afft/radix/difreq_compound_radix2.hpp"
#include "afft/radix/difreq_compound_radix4.hpp"
#include "afft/radix/difreq_radix2.hpp"
#include "afft/radix/difreq_radix4.hpp"
#include "afft/radix/ditime_compound_radix2.hpp"
#include "afft/radix/ditime_compound_radix4.hpp"
#include "afft/radix/ditime_radix2.hpp"
#include "afft/radix/ditime_radix4.hpp"
#include "afft/radix/radix_stage.hpp"
#include "afft/radix/radix_type.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class ButterflyImpl
    {
        using SampleSpec = typename BoundedSpec<Spec, 0>::spec;

        ButterflyPlan<SampleSpec, Allocator> plan_;

        public:
        explicit ButterflyImpl(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand) {}

        const ButterflyPlan<SampleSpec, Allocator> &plan() const
        {
            return plan_;
        }

        template <bool Rescaling = false>
        static inline void eval_ditime(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag,
            const ButterflyPlan<SampleSpec, Allocator> &plan)
        {

            using sample = typename Spec::sample;
            sample scaling_factor;

            if (Rescaling){
                scaling_factor = plan.scaling_factor();
            }
            else {
                scaling_factor = sample(1);
            }

            for (const auto &radix_stage : plan.radix_stages())
            {
                switch (radix_stage.type)
                {
                case RadixType::radix4:
                case RadixType::carry_radix4:
                    do_ditime_radix4_stage<Spec>(
                        out_real,
                        out_imag,
                        out_real,
                        out_imag,
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

                case RadixType::radix2:
                case RadixType::carry_radix2:
                    do_ditime_radix2_stage<Spec>(
                        out_real,
                        out_imag,
                        out_real,
                        out_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subtwiddle_len,
                        radix_stage.subtwiddle_start,
                        radix_stage.subtwiddle_end);
                    break;

                case RadixType::compound_radix4:
                case RadixType::carry_compound_radix4:
                    do_ditime_compound_radix4_stage<Spec, Rescaling>(
                        out_real,
                        out_imag,
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        scaling_factor);
                    break;

                case RadixType::compound_radix2:
                    do_ditime_compound_radix2_stage<Spec, Rescaling>(
                        out_real,
                        out_imag,
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        scaling_factor);
                    break;
                }
            }
        }

        template <bool Rescaling = false>
        static inline void eval_difreq(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag,
            const ButterflyPlan<SampleSpec, Allocator> &plan)
        {

            using sample = typename Spec::sample;
            sample scaling_factor;

            if (Rescaling){
                scaling_factor = plan.scaling_factor();
            }
            else {
                scaling_factor = sample(1);
            }
            const auto &radix_stages = plan.radix_stages();
            for (std::size_t stage_id = radix_stages.size(); stage_id-- > 0;)
            {
                const auto &radix_stage = radix_stages[stage_id];
                switch (radix_stage.type)
                {
                case RadixType::radix4:
                    do_difreq_radix4_stage<Spec>(
                        out_real,
                        out_imag,
                        out_real,
                        out_imag,
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

                case RadixType::carry_radix4:
                    do_difreq_radix4_stage<Spec>(
                        out_real,
                        out_imag,
                        in_real,
                        in_imag,
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

                case RadixType::radix2:
                    do_difreq_radix2_stage<Spec>(
                        out_real,
                        out_imag,
                        out_real,
                        out_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subtwiddle_len,
                        radix_stage.subtwiddle_start,
                        radix_stage.subtwiddle_end);
                    break;

                case RadixType::carry_radix2:
            
                    do_difreq_radix2_stage<Spec>(
                        out_real,
                        out_imag,
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subtwiddle_len,
                        radix_stage.subtwiddle_start,
                        radix_stage.subtwiddle_end);
                    break;

                case RadixType::compound_radix4:
                    do_difreq_compound_radix4_stage<Spec, Rescaling>(
                        out_real,
                        out_imag,
                        out_real,
                        out_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        scaling_factor);
                    break;
                case RadixType::carry_compound_radix4:
                    do_difreq_compound_radix4_stage<Spec, Rescaling>(
                        out_real,
                        out_imag,
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        radix_stage.subfft_id_start,
                        radix_stage.subfft_id_end,
                        scaling_factor);
                    break;

                case RadixType::compound_radix2:
                    do_difreq_compound_radix2_stage<Spec, Rescaling>(
                        out_real,
                        out_imag,
                        in_real,
                        in_imag,
                        radix_stage.tw_real_b,
                        radix_stage.tw_imag_b,
                        scaling_factor);
                    break;
                }
            }
        }

        template <bool Rescaling = false>
        void eval_ditime(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            eval_ditime<Rescaling>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                plan_);
        }

        template <bool Rescaling = false>
        void eval_difreq(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            eval_difreq<Rescaling>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                plan_);
        }
    };
}

#endif