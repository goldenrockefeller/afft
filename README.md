# afft
Adequately Fast Fourier Transform

This is supposed to be a small C++ project to make a fast, template-based implementaiton of mixed radix (2 and 4) FFT (Fast Fourier Transform). AFFT will have the ability to be extended with SIMD instructions like SSE and AVX for even more speed.  The goal is to use this code in future audio dsp project. 

## Goals
- C++11
- Highly portable, yet efficient
- Template-based for easy adaptation to other platforms and future-proofing (extends to AVX and Neon)
- 
- Power of 2 (targeting under 2^22)
- Cache-Oblivious

## Implementation
- Expects different arrays for real and imaginary numbers (not directly supporting interweaved input, not direction support array or complex numbers)
- Cooley-Tuker
- 
