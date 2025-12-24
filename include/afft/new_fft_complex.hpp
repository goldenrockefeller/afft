#ifndef AFFT_NEW_FFT_COMPLEX_HPP
#define AFFT_NEW_FFT_COMPLEX_HPP

#include <vector>
#include <unordered_map>
#include <cstddef>
#include <iostream>

#include "afft/stage/stage.hpp"
#include "afft/plan/plan_fft_complex.hpp"
#include "afft/common_math.hpp"
#include "afft/operations/s_radix2.hpp"
#include "afft/operations/s_radix4.hpp"
#include "afft/operations/ct_radix2.hpp"
#include "afft/operations/ct_radix4.hpp"
#include "afft/stage/stage.hpp"
#include "afft/stage/stage_type.hpp"
#include "afft/spec/bounded_spec.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class NewFftComplex
    {
    public:
        using sample_spec = typename BoundedSpec<Spec, 0>::spec;
        using sample = typename Spec::sample; 

    private:
        enum data_ids { in_real = 0, in_imag = 1, out_real = 2, out_imag = 3, buf_real = 4, buf_imag = 5 };

        std::vector<Stage<sample>> plan_;
        std::vector<Stage<sample>> scaled_plan_;
        std::vector<sample, Allocator> twiddles_;
        std::vector<std::size_t> in_permute_indexes_;
        std::vector<std::size_t> out_permute_indexes_;
        std::size_t n_samples_per_operand_;
        std::size_t log_n_samples_per_operand_;
        std::size_t n_samples_;
        sample scaling_factor_;

        mutable std::vector<sample, Allocator> buf_;    

    public:

        explicit NewFftComplex(std::size_t n_samples) 
        {
            namespace cm = afft::common_math;
            const std::size_t max_n_samples_per_operand =  Spec::n_samples_per_operand;

            n_samples_ = n_samples;


            // Compute n_samples_per_operand as in the plan function
            auto log_n_samples_per_operand =
                std::min(
                    cm::int_log_2(max_n_samples_per_operand),
                    std::size_t(cm::int_log_2(n_samples / 2)));
            std::size_t n_samples_per_operand = 1 << log_n_samples_per_operand;

            log_n_samples_per_operand_ = log_n_samples_per_operand;
            n_samples_per_operand_ = n_samples_per_operand;

            
            // Create the butterfly plan
            plan_ = plan::complex_fft_plan<sample>(n_samples, n_samples_per_operand, Spec::min_partition_len);

            // Get twiddles
            auto twiddles_result = plan::twiddles<sample_spec, Allocator>(plan_, n_samples, n_samples_per_operand);
            twiddles_ = std::move(twiddles_result.first);
            auto twiddle_map = std::move(twiddles_result.second);

            // Get bit reverse indexes
            auto bit_reverse_result = plan::bit_reverse_indexes(plan_, n_samples, n_samples_per_operand);
            in_permute_indexes_ = std::move(bit_reverse_result.first);
            out_permute_indexes_ = std::move(bit_reverse_result.second);

            // Set the pointers in the plan
            plan::set_twiddle_pointers(plan_, twiddle_map);
            plan::set_bit_reverse_pointers(plan_, in_permute_indexes_, out_permute_indexes_);

            // Set s_radix ids: in_real=0, in_imag=0 (dummy), out_real=1, out_imag=1, buf_real=0, buf_imag=0
            plan::set_data_ids_for_complex_fft(
                plan_, 
                data_ids::in_real,
                data_ids::in_imag,
                data_ids::out_real, 
                data_ids::out_imag, 
                data_ids::buf_real,
                data_ids::buf_imag
            );

            // Create scaled_plan_ by replacing _init stages with _init_rescale counterparts
            scaled_plan_ = plan_;
            for (auto& stage : scaled_plan_) {
                if (stage.type == StageType::s_radix4_init) {
                    stage.type = StageType::s_radix4_init_rescale;
                } else if (stage.type == StageType::s_radix2_init) {
                    stage.type = StageType::s_radix2_init_rescale;
                }
            }

            scaling_factor_ = sample(1) / sample(n_samples);

            buf_.resize(n_samples * 2);
            
        }

        NewFftComplex(const NewFftComplex&) = default;
        NewFftComplex(NewFftComplex&&) = default;
        NewFftComplex& operator=(const NewFftComplex&) = default;
        NewFftComplex& operator=(NewFftComplex&&) = default;
        ~NewFftComplex() = default;

        const std::vector<Stage<sample>>& plan() const
        {
            return plan_;
        }

        const std::vector<Stage<sample>>& scaled_plan() const
        {
            return scaled_plan_;
        }

        const std::vector<sample>& twiddles() const
        {
            return twiddles_;
        }

        const std::vector<std::size_t>& in_permute_indexes() const
        {
            return in_permute_indexes_;
        }

        const std::vector<std::size_t>& out_permute_indexes() const
        {
            return out_permute_indexes_;
        }

        std::size_t n_samples_per_operand() const
        {
            return n_samples_per_operand_;
        }

        std::size_t n_samples() const
        {
            return n_samples_;
        }

        std::size_t log_n_samples_per_operand() const
        {
            return log_n_samples_per_operand_;
        }

        const sample &scaling_factor() const
        {
            return scaling_factor_;
        }


        template <typename BoundedSpec>
        static inline void bounded_eval(
            sample **data,
            std::size_t n_samples,
            const sample &scaling_factor,
            const std::vector<Stage<sample>> &plan)
        {
            if (plan.empty()) return;


            for (std::size_t stage_id = 0; stage_id < plan.size(); stage_id++)
            {
                const auto &stage = plan[stage_id];

                switch (stage.type)
                {
                case StageType::ct_radix4:
                    {
                        auto &params = stage.params.ct_r4;
                        do_ct_radix4_stage<BoundedSpec>(
                            data[params.inout_real_id],
                            data[params.inout_imag_id],
                            params.twiddles,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.subtwiddle_len,
                            params.subtwiddle_start,
                            params.subtwiddle_end);
                    }
                    break;
                
                case StageType::ct_radix2:
                    {
                        auto &params = stage.params.ct_r2;
                        
                        do_ct_radix2_stage<BoundedSpec>(
                            data[params.inout_real_id],
                            data[params.inout_imag_id],
                            params.twiddles,
                            params.subtwiddle_len,
                            params.subtwiddle_start,
                            params.subtwiddle_end);
                    }
                    break;
                case StageType::s_radix4:
                    {
                        auto &params = stage.params.s_r;
                        do_s_radix4_stage<BoundedSpec, false, true>(
                            data[params.out_real_id],
                            data[params.out_imag_id],
                            data[params.in_real_id],
                            data[params.in_imag_id],
                            params.twiddles,
                            params.out_permute_indexes,
                            params.in_permute_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            n_samples,
                            sample(1)
                        );
                            
                    }
                    break;
                case StageType::s_radix2:
                    {
                        auto &params = stage.params.s_r;
                        do_s_radix2_stage<BoundedSpec, false, true>(
                            data[params.out_real_id],
                            data[params.out_imag_id],
                            data[params.in_real_id],
                            data[params.in_imag_id],
                            params.twiddles,
                            params.out_permute_indexes,
                            params.in_permute_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            n_samples,
                            sample(1)
                        );
                    }
                    break;
                case StageType::s_radix4_init:
                    {
                        auto &params = stage.params.s_r;
                        do_s_radix4_stage<BoundedSpec, false, false>(
                            data[params.out_real_id],
                            data[params.out_imag_id],
                            data[params.in_real_id],
                            data[params.in_imag_id],
                            params.twiddles,
                            params.out_permute_indexes,
                            params.in_permute_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            n_samples,
                            sample(1)
                        );
                    }
                    break;
                case StageType::s_radix2_init:
                    {
                        auto &params = stage.params.s_r;
                        do_s_radix2_stage<BoundedSpec, false, false>(
                            data[params.out_real_id],
                            data[params.out_imag_id],
                            data[params.in_real_id],
                            data[params.in_imag_id],
                            params.twiddles,
                            params.out_permute_indexes,
                            params.in_permute_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            n_samples,
                            sample(1)
                        );
                    }
                    break;
                case StageType::s_radix4_init_rescale:
                    {
                        auto &params = stage.params.s_r;
                        do_s_radix4_stage<BoundedSpec, true, false>(
                            data[params.out_real_id],
                            data[params.out_imag_id],
                            data[params.in_real_id],
                            data[params.in_imag_id],
                            params.twiddles,
                            params.out_permute_indexes,
                            params.in_permute_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            n_samples,
                            scaling_factor);
                    }
                    break;
                case StageType::s_radix2_init_rescale:
                    {
                        auto &params = stage.params.s_r;
                        do_s_radix2_stage<BoundedSpec, true, false>(
                            data[params.out_real_id],
                            data[params.out_imag_id],
                            data[params.in_real_id],
                            data[params.in_imag_id],
                            params.twiddles,
                            params.out_permute_indexes,
                            params.in_permute_indexes,
                            params.subfft_id_start,
                            params.subfft_id_end,
                            params.log_interleave_permute,
                            n_samples,
                            scaling_factor);
                    }
                    break;
                default:
                    break;
                }
            }
        }

        static inline void eval(
            sample **data,
            std::size_t n_samples,
            std::size_t log_n_samples_per_operand,
            const sample &scaling_factor,
            const std::vector<Stage<sample>> &plan)
        {

            switch (log_n_samples_per_operand)
            {
            case 0:
                bounded_eval<typename BoundedSpec<Spec, 0>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 1:
                bounded_eval<typename BoundedSpec<Spec, 1>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 2:
                bounded_eval<typename BoundedSpec<Spec, 2>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 3:
                bounded_eval<typename BoundedSpec<Spec, 3>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 4:
                bounded_eval<typename BoundedSpec<Spec, 4>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 5:
                bounded_eval<typename BoundedSpec<Spec, 5>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 6:
                bounded_eval<typename BoundedSpec<Spec, 6>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 7:
                bounded_eval<typename BoundedSpec<Spec, 7>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            case 8:
                bounded_eval<typename BoundedSpec<Spec, 8>::spec>(
                    data,
                    n_samples,
                    scaling_factor,
                    plan);
                break;
            default:
                break;
            }
        }

        void eval(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            sample *data[6] = {
                const_cast<sample*>(in_real),
                const_cast<sample*>(in_imag),
                out_real,
                out_imag,
                buf_.data(),
                buf_.data() + n_samples_
            };

            eval(
                data,
                n_samples_,
                log_n_samples_per_operand_,
                scaling_factor_,
                plan_);
        }

        void fft(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            sample *data[6] = {
                const_cast<sample*>(in_real),
                const_cast<sample*>(in_imag),
                out_real,
                out_imag,
                buf_.data(),
                buf_.data() + n_samples_
            };

            eval(
                data,
                n_samples_,
                log_n_samples_per_operand_,
                scaling_factor_,
                plan_);
        }

        void fft_normalized(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            sample *data[6] = {
                const_cast<sample*>(in_real),
                const_cast<sample*>(in_imag),
                out_real,
                out_imag,
                buf_.data(),
                buf_.data() + n_samples_
            };

            eval(
                data,
                n_samples_,
                log_n_samples_per_operand_,
                scaling_factor_,
                scaled_plan_);
        }

        void ifft(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            sample *data[6] = {
                out_real,
                out_imag,
                const_cast<sample*>(in_real),
                const_cast<sample*>(in_imag),
                buf_.data(),
                buf_.data() + n_samples_
            };

            eval(
                data,
                n_samples_,
                log_n_samples_per_operand_,
                scaling_factor_,
                plan_);
        }

        void ifft_normalized(
            sample *out_real,
            sample *out_imag,
            const sample *in_real,
            const sample *in_imag) const
        {
            sample *data[6] = {
                out_real,
                out_imag,
                const_cast<sample*>(in_real),
                const_cast<sample*>(in_imag),
                buf_.data(),
                buf_.data() + n_samples_
            };

            eval(
                data,
                n_samples_,
                log_n_samples_per_operand_,
                scaling_factor_,
                scaled_plan_);
        }
    };
}

#endif