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
        mutable std::vector<sample, Allocator> buf_real_;
        mutable std::vector<sample, Allocator> buf_imag_;        
        mutable std::vector<sample, Allocator> buf_real_2_;
        mutable std::vector<sample, Allocator> buf_imag_2_;


    public:
        explicit Butterfly(std::size_t n_samples) : plan_(n_samples, Spec::n_samples_per_operand), buf_real_(n_samples), buf_imag_(n_samples), buf_real_2_(n_samples), buf_imag_2_(n_samples) {}
        Butterfly(std::size_t n_samples, std::size_t min_partition_len) : plan_(n_samples, Spec::n_samples_per_operand, min_partition_len), buf_real_(n_samples), buf_imag_(n_samples), buf_real_2_(n_samples), buf_imag_2_(n_samples) {}

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
            sample *buf_real,
            sample *buf_imag,
            sample *buf_real_2,
            sample *buf_imag_2,
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
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 1:
                ButterflyImpl<typename BoundedSpec<Spec, 1>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 2:
                ButterflyImpl<typename BoundedSpec<Spec, 2>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 3:
                ButterflyImpl<typename BoundedSpec<Spec, 3>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 4:
                ButterflyImpl<typename BoundedSpec<Spec, 4>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 5:
                ButterflyImpl<typename BoundedSpec<Spec, 5>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 6:
                ButterflyImpl<typename BoundedSpec<Spec, 6>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 7:
                ButterflyImpl<typename BoundedSpec<Spec, 7>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
                    plan);
                break;
            case 8: // Maximum support number of samples per operand is 256!
                ButterflyImpl<typename BoundedSpec<Spec, 8>::spec, Allocator>::template eval<Rescaling>(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    buf_real,
                    buf_imag,
                    buf_real_2,
                    buf_imag_2,
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
                buf_real_.data(),
                buf_imag_.data(),
                buf_real_2_.data(),
                buf_imag_2_.data(),
                plan_);
        }
    };
}

#endif