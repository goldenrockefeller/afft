#ifndef AFFT_EXECUTE_HPP
#define AFFT_EXECUTE_HPP

#include <vector>
#include <cstddef>

#include "afft/stage/stage.hpp"
#include "afft/stage/stage_type.hpp"
#include "afft/spec/bounded_spec.hpp"
#include "afft/operations/s_radix2.hpp"
#include "afft/operations/s_radix4.hpp"
#include "afft/operations/ct_radix2.hpp"
#include "afft/operations/ct_radix4.hpp"

namespace afft
{
    namespace execute
    {
        template <typename BoundedSpec>
        inline void bounded_eval(
            typename BoundedSpec::sample **data,
            const std::vector<Stage<typename BoundedSpec::sample>> &plan)
        {
            using sample = typename BoundedSpec::sample;
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
                            params.n_samples,
                            sample(1),
                            params.log_interleave_permute
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
                            params.n_samples,
                            sample(1),
                            params.log_interleave_permute
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
                            params.n_samples,
                            sample(1),
                            params.log_interleave_permute
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
                            params.n_samples,
                            sample(1),
                            params.log_interleave_permute
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
                            params.n_samples,
                            params.scaling_factor,
                            params.log_interleave_permute
                            );
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
                            params.n_samples,
                            params.scaling_factor,
                            params.log_interleave_permute);
                    }
                    break;
                default:
                    break;
                }
            }
        }

        template <typename Spec>
        inline void eval(
            typename Spec::sample **data,
            const std::vector<Stage<typename Spec::sample>> &plan,
            std::size_t log_n_samples_per_operand)
        {
            switch (log_n_samples_per_operand)
            {
            case 0:
                bounded_eval<typename BoundedSpec<Spec, 0>::spec>(
                    data,
                    plan);
                break;
            case 1:
                bounded_eval<typename BoundedSpec<Spec, 1>::spec>(
                    data,
                    plan);
                break;
            case 2:
                bounded_eval<typename BoundedSpec<Spec, 2>::spec>(
                    data,
                    plan);
                break;
            case 3:
                bounded_eval<typename BoundedSpec<Spec, 3>::spec>(
                    data,
                    plan);
                break;
            case 4:
                bounded_eval<typename BoundedSpec<Spec, 4>::spec>(
                    data,
                    plan);
                break;
            case 5:
                bounded_eval<typename BoundedSpec<Spec, 5>::spec>(
                    data,
                    plan);
                break;
            case 6:
                bounded_eval<typename BoundedSpec<Spec, 6>::spec>(
                    data,
                    plan);
                break;
            case 7:
                bounded_eval<typename BoundedSpec<Spec, 7>::spec>(
                    data,
                    plan);
                break;
            case 8:
                bounded_eval<typename BoundedSpec<Spec, 8>::spec>(
                    data,
                    plan);
                break;
            default:
                break;
            }
        }
    }
}

#endif
