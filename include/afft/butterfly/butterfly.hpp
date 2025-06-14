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
        using sample_spec = typename BoundedSpec<Spec, 0>::spec;
        using sample = typename Spec::sample;

        ButterflyPlan<sample_spec, Allocator> plan_;
        mutable std::vector<sample, Allocator> buf_;      


    public:
        explicit Butterfly(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand, Spec::prefetch_lookahead, Spec::min_partition_len), buf_(2 * n_samples + 2048) {}
        
        const ButterflyPlan<sample_spec, Allocator> &plan() const
        {
            return plan_;
        }

        template <bool Rescaling = false>
        static inline void eval(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag,
            sample *buf,
            const ButterflyPlan<sample_spec, Allocator> &plan)
        {
            switch (plan.log_n_samples_per_operand())
            { //
            case 0:
                ButterflyImpl<typename BoundedSpec<Spec, 0>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 1:
                ButterflyImpl<typename BoundedSpec<Spec, 1>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 2:
                ButterflyImpl<typename BoundedSpec<Spec, 2>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 3:
                ButterflyImpl<typename BoundedSpec<Spec, 3>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 4:
                ButterflyImpl<typename BoundedSpec<Spec, 4>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 5:
                ButterflyImpl<typename BoundedSpec<Spec, 5>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 6:
                ButterflyImpl<typename BoundedSpec<Spec, 6>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 7:
                ButterflyImpl<typename BoundedSpec<Spec, 7>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            case 8: // Maximum support number of samples per operand is 256!
                ButterflyImpl<typename BoundedSpec<Spec, 8>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf,
                    plan);
                break;
            }
        }

        template <bool Rescaling = false>
        void eval(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            eval<Rescaling>(
                out_real,
                out_imag,
                in_real,
                in_imag,
                buf_.data(),
                plan_);
        }
    };
}

#endif