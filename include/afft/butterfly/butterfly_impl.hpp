#ifndef AFFT_BUTTERFLY_IMPL_HPP
#define AFFT_BUTTERFLY_IMPL_HPP

#include <cstddef>
#include <vector>
#include <cstring>
#include "afft/butterfly/butterfly_plan.hpp"
#include "afft/radix/difreq_s_radix2.hpp"
#include "afft/radix/difreq_s_radix4.hpp"
#include "afft/radix/difreq_ct_radix2.hpp"
#include "afft/radix/difreq_ct_radix4.hpp"
#include "afft/radix/ditime_s_radix2.hpp"
#include "afft/radix/ditime_s_radix4.hpp"
#include "afft/radix/ditime_ct_radix2.hpp"
#include "afft/radix/ditime_ct_radix4.hpp"
#include "afft/radix/radix_stage.hpp"
#include "afft/radix/radix_type.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class ButterflyImpl
    {
        using SampleSpec = typename BoundedSpec<Spec, 0>::spec;
        using sample = typename Spec::sample; 

        ButterflyPlan<SampleSpec, Allocator> plan_;
        mutable std::vector<sample, Allocator> buf_real_;
        mutable std::vector<sample, Allocator> buf_imag_;

        public:
        explicit ButterflyImpl(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand), buf_real_(n_samples), buf_imag_(n_samples) {}
        ButterflyImpl(std::size_t n_samples, std::size_t min_partition_len) : plan_(n_samples, Spec::n_samples_per_operand, min_partition_len), buf_real_(n_samples), buf_imag_(n_samples)  {}

        const ButterflyPlan<SampleSpec, Allocator> &plan() const
        {
            return plan_;
        }

        template <bool Rescaling = false>
        static inline void eval_ditime(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag,
            sample *buf_real,
            sample *buf_imag,
            const ButterflyPlan<SampleSpec, Allocator> &plan)
        {

            using sample = sample;
            sample scaling_factor;

            if (Rescaling){
                scaling_factor = plan.scaling_factor();
            }
            else {
                scaling_factor = sample(1);
            }

            bool on_first_stage = true;
            
            if (plan.n_s_radix_stages() % 2 == 1)
            {
                std::swap(out_real, buf_real);
                std::swap(out_imag, buf_imag);
            }

            if (plan.n_s_radix_stages() == 1)
            {
                std::swap(out_real, buf_real);
                std::swap(out_imag, buf_imag);
            }

            for (const auto &radix_stage : plan.radix_stages())
            {
                if (!on_first_stage) {
                    switch (radix_stage.type)
                    {
                    case RadixType::ct_radix4:
                        do_ditime_ct_radix4_stage<Spec, Rescaling>(
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
                    case RadixType::s_radix4:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_ditime_s_radix4_stage<Spec, Rescaling>(
                            out_real,
                            out_imag,
                            buf_real,
                            buf_imag,
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
                            radix_stage.log_subtwiddle_len);
                        break;
                    case RadixType::ct_radix2:
                        do_ditime_ct_radix2_stage<Spec>(
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
                    case RadixType::s_radix2:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_ditime_s_radix2_stage<Spec, Rescaling>(
                            out_real,
                            out_imag,
                            buf_real,
                            buf_imag,
                            radix_stage.tw_real_b,
                            radix_stage.tw_imag_b,
                            radix_stage.out_indexes,
                            radix_stage.in_indexes,
                            radix_stage.subfft_id_start,
                            radix_stage.subfft_id_end,
                            radix_stage.log_subtwiddle_len);
                        break;
                    }
                }
                else {
                    switch (radix_stage.type)
                    {
                    case RadixType::ct_radix4:
                        break;
                    case RadixType::s_radix4:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_ditime_s_radix4_stage<Spec, Rescaling>(
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
                            radix_stage.out_indexes,
                            radix_stage.in_indexes,
                            radix_stage.subfft_id_start,
                            radix_stage.subfft_id_end,
                            radix_stage.log_subtwiddle_len);
                        break;
                    case RadixType::ct_radix2:
                        break;
                    case RadixType::s_radix2:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_ditime_s_radix2_stage<Spec, Rescaling>(
                            out_real,
                            out_imag,
                            in_real,
                            in_imag,
                            radix_stage.tw_real_b,
                            radix_stage.tw_imag_b,
                            radix_stage.out_indexes,
                            radix_stage.in_indexes,
                            radix_stage.subfft_id_start,
                            radix_stage.subfft_id_end,
                            radix_stage.log_subtwiddle_len);
                        break;
                    }

                    on_first_stage = false;
                }
            }

            if (plan.n_s_radix_stages() == 1)
            {
                std::memcpy(out_real, buf_real, sizeof(sample) * plan.n_samples());
                std::memcpy(out_imag, buf_imag, sizeof(sample) * plan.n_samples());
            }
        }

        template <bool Rescaling = false>
        static inline void eval_difreq(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag,
            sample *buf_real,
            sample *buf_imag,
            const ButterflyPlan<SampleSpec, Allocator> &plan)
        {

            using sample = sample;
            sample scaling_factor;

            if (Rescaling){
                scaling_factor = plan.scaling_factor();
            }
            else {
                scaling_factor = sample(1);
            }

            bool on_first_stage = true;
            
            if (plan.n_s_radix_stages() % 2 == 1)
            {
                std::swap(out_real, buf_real);
                std::swap(out_imag, buf_imag);
            }

            if (plan.n_s_radix_stages() == 1)
            {
                std::swap(out_real, buf_real);
                std::swap(out_imag, buf_imag);
            }


            for (std::size_t radix_stage_id = plan.radix_stages(); radix_stage_id-- > 0;)
            {
                const auto &radix_stage = plan.radix_stages()[radix_stage_id];
                if  (!on_first_stage) {
                    switch (radix_stage.type)
                    {
                    case RadixType::ct_radix4:
                        do_difreq_ct_radix4_stage<Spec, Rescaling>(
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
                    case RadixType::s_radix4:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_difreq_s_radix4_stage<Spec, Rescaling>(
                            out_real,
                            out_imag,
                            buf_real,
                            buf_imag,
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
                            radix_stage.log_subtwiddle_len);
                        break;
                    case RadixType::ct_radix2:
                        do_difreq_ct_radix2_stage<Spec>(
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
                    case RadixType::s_radix2:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_difreq_s_radix2_stage<Spec, Rescaling>(
                            out_real,
                            out_imag,
                            buf_real,
                            buf_imag,
                            radix_stage.tw_real_b,
                            radix_stage.tw_imag_b,
                            radix_stage.out_indexes,
                            radix_stage.in_indexes,
                            radix_stage.subfft_id_start,
                            radix_stage.subfft_id_end,
                            radix_stage.log_subtwiddle_len);
                        break;
                    }
                }
                else {
                    switch (radix_stage.type)
                    {
                    case RadixType::ct_radix4:
                        do_difreq_ct_radix4_stage<Spec, Rescaling>(
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
                    case RadixType::s_radix4:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_difreq_s_radix4_stage<Spec, Rescaling>(
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
                            radix_stage.out_indexes,
                            radix_stage.in_indexes,
                            radix_stage.subfft_id_start,
                            radix_stage.subfft_id_end,
                            radix_stage.log_subtwiddle_len);
                        break;
                    case RadixType::ct_radix2:
                        do_difreq_ct_radix2_stage<Spec>(
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
                    case RadixType::s_radix2:
                        std::swap(out_real, buf_real);
                        std::swap(out_imag, buf_imag);
                        do_difreq_s_radix2_stage<Spec, Rescaling>(
                            out_real,
                            out_imag,
                            in_real,
                            in_imag,
                            radix_stage.tw_real_b,
                            radix_stage.tw_imag_b,
                            radix_stage.out_indexes,
                            radix_stage.in_indexes,
                            radix_stage.subfft_id_start,
                            radix_stage.subfft_id_end,
                            radix_stage.log_subtwiddle_len);
                        break;
                    }

                    on_first_stage = false;
                }
            }

            if (plan.n_s_radix_stages() == 1)
            {
                std::memcpy(out_real, buf_real, sizeof(sample) * plan.n_samples());
                std::memcpy(out_imag, buf_imag, sizeof(sample) * plan.n_samples());
            }
        }

        template <bool Rescaling = false>
        void eval_ditime(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            eval_ditime<Rescaling>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                buf_real_.data(),
                buf_imag_.data(),
                plan_);
        }

        template <bool Rescaling = false>
        void eval_difreq(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            eval_difreq<Rescaling>(
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