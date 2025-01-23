# afft
Adequately Fast Fourier Transform

This is supposed to be a small C++ project to make a fast, template-based implementaiton of mixed radix (2 and 4) FFT (Fast Fourier Transform), mainly for audio dsp. AFFT will have the ability to be extended with SIMD instructions like SSE and AVX for even more speed.  The goal is to use this code in future audio dsp project. 

Prototype

## Goals
- C++11
- Highly portable, yet efficient
- Template-based for easy adaptation to other platforms and future-proofing (extends to AVX and Neon)
- Header-only
- Power of 2 (targeting under 2^22)
- Cache-Oblivious where possible
- Liberal license

## Inspiration
- PFFFT
- PGFFT
- OTFFT
- Ryg's Blog on FFT impentation https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/
- Prototype

## Implementation
- Expects different arrays for real and imaginary numbers (not directly supporting interweaved input, not direction support array or complex numbers)
- Cooley-Tukey (mixed radix of 2 and 4)
- 

## Investigated
- While not benchmarked, Radix-8 and split radix possibly not worth the effort.
  - For radix-4: the number of operations for sample is (3 twiddles complex multiplies (x6) + 8 complex additions (x2) for 4 complex numbers simulating 2 radix-2 stages) = 4.25 real operations per complex sample
  - For radix-8: using special reduction for 1+i: (7 twiddles complex multiplies (x6) + 24 complex additions (x2) + 4 "1+i" multiplication (x2) for 8 complex numbers simulating 3 radix-2 stages) =  4.083
  - Theoretically, radix-8 gives a only 4% decrease in time, but this reduction might be less with we consider that the more operations can more easily overload registers. 
- - With FMA having the same through put of a single add or Muli
- Unrolling the main radix-4 and radix-2 loops does not give much speed up.
- Compiling with Clang gives 5% to 20% speed up over compiling with MSVC or GCC (on Windows, Intel i7-12700)
- 
## Investigating


