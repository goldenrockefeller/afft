#ifndef AFFT_BOUNDED_SPEC_HPP
#define AFFT_BOUNDED_SPEC_HPP


namespace afft
{
    template <typename Spec, std::size_t MaxLogNSamplesPerOperand, bool WithinBounds>
    struct CheckedBoundedSpec{
        using spec = typename CheckedBoundedSpec<typename Spec::fallback_spec, MaxLogNSamplesPerOperand, Spec::n_samples_per_operand / 2 <= (1 << MaxLogNSamplesPerOperand)>::spec;
    };

    template <typename Spec, std::size_t MaxLogNSamplesPerOperand>
    struct CheckedBoundedSpec<Spec, MaxLogNSamplesPerOperand, true>{
        using spec = Spec;
    };

    template <typename Spec, std::size_t MaxLogNSamplesPerOperand>
    struct BoundedSpec{
        using spec = typename CheckedBoundedSpec<Spec, MaxLogNSamplesPerOperand, Spec::n_samples_per_operand <= (1 << MaxLogNSamplesPerOperand)>::spec;
    };
} // namespace afft


#endif