#ifndef AFFT_BUTTERFLY_HPP
#define AFFT_BUTTERFLY_HPP

#include <cstddef>
#include <vector>
#include "afft/butterfly/butterfly_plan.hpp"
#include "afft/butterfly/butterfly_impl.hpp"
#include "afft/spec/bounded_spec.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class Butterfly
    {
        using SampleSpec = typename BoundedSpec<Spec, 0>::spec;

        ButterflyPlan<SampleSpec, Allocator> plan_;

    public:
        explicit Butterfly(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand) {}
        Butterfly(std::size_t n_samples, std::size_t min_partition_len) : plan_(n_samples, Spec::n_samples_per_operand, min_partition_len) {}

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
    
            switch (plan.log_n_samples_per_operand())
            { //
            case LogNSamplesPerOperand::n0:
                ButterflyImpl<typename BoundedSpec<Spec, 0>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n1:
                ButterflyImpl<typename BoundedSpec<Spec, 1>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n2:
                ButterflyImpl<typename BoundedSpec<Spec, 2>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n3:
                ButterflyImpl<typename BoundedSpec<Spec, 3>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n4:
                ButterflyImpl<typename BoundedSpec<Spec, 4>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n5:
                ButterflyImpl<typename BoundedSpec<Spec, 5>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n6:
                ButterflyImpl<typename BoundedSpec<Spec, 6>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n7:
                ButterflyImpl<typename BoundedSpec<Spec, 7>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n8: // Maximum support number of samples per operand is 256!
                ButterflyImpl<typename BoundedSpec<Spec, 8>::spec, Allocator>::template eval_ditime<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
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
    
            switch (plan.log_n_samples_per_operand())
            { //
            case LogNSamplesPerOperand::n0:
                ButterflyImpl<typename BoundedSpec<Spec, 0>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n1:
                ButterflyImpl<typename BoundedSpec<Spec, 1>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n2:
                ButterflyImpl<typename BoundedSpec<Spec, 2>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n3:
                ButterflyImpl<typename BoundedSpec<Spec, 3>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n4:
                ButterflyImpl<typename BoundedSpec<Spec, 4>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n5:
                ButterflyImpl<typename BoundedSpec<Spec, 5>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n6:
                ButterflyImpl<typename BoundedSpec<Spec, 6>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n7:
                ButterflyImpl<typename BoundedSpec<Spec, 7>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case LogNSamplesPerOperand::n8: // Maximum support number of samples per operand is 256!
                ButterflyImpl<typename BoundedSpec<Spec, 8>::spec, Allocator>::template eval_difreq<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
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