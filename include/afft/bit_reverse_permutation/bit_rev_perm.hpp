#ifndef AFFT_BIT_REV_PERM_HPP
#define AFFT_BIT_REV_PERM_HPP

#include <cstddef>
#include <vector>
#include "afft/bit_reverse_permutation/bit_rev_perm_plan.hpp"
#include "afft/bit_reverse_permutation/bit_rev_perm_impl.hpp"
#include "afft/spec/bounded_spec.hpp"

namespace afft
{
    template <typename Spec>
    class BitRevPerm
    {
        BitRevPermPlan plan_;

    public:
        explicit BitRevPerm(std::size_t n_indexes) : plan_(n_indexes, Spec::n_samples_per_operand) {}

        const BitRevPermPlan &plan() const
        {
            return plan_;
        }

        static void eval(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag,
            const BitRevPermPlan &plan)
        {
    
            switch (plan.log_n_sample_per_operand())
            { //
            case 0:
                BitRevPermImpl<typename BoundedSpec<Spec, 0>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 1:
                BitRevPermImpl<typename BoundedSpec<Spec, 1>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 2:
                BitRevPermImpl<typename BoundedSpec<Spec, 2>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 3:
                BitRevPermImpl<typename BoundedSpec<Spec, 3>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 4:
                BitRevPermImpl<typename BoundedSpec<Spec, 4>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 5:
                BitRevPermImpl<typename BoundedSpec<Spec, 5>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 6:
                BitRevPermImpl<typename BoundedSpec<Spec, 6>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 7:
                BitRevPermImpl<typename BoundedSpec<Spec, 7>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            case 8: // Maximum support number of samples per operand is 256!
                BitRevPermImpl<typename BoundedSpec<Spec, 8>::spec>::eval(
                    out_real,
                    out_imag,
                    in_real,
                    in_imag,
                    plan);
                break;
            }
        }

        void eval(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag)
        {
            eval(
                out_real,
                out_imag,
                in_real,
                in_imag,
                plan_);
        }
    };
}

#endif