#ifndef AFFT_RADIX_TYPE_HPP
#define AFFT_RADIX_TYPE_HPP

namespace afft{
    enum class RadixType {
        radix4,
        carry_radix4,
        radix2,
        compound_radix4,
        carry_compound_radix4,
        compound_radix2
    };
}

#endif