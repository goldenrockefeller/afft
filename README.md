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

## Implementation
- Expects different arrays for real and imaginary numbers (not directly supporting interweaved input, not direction support array or complex numbers)
- Cooley-Tukey (mixed radix of 2 and 4)
- Possible to add SIMD-enabled bit-reversal (implement for AVX-double for now)
  
## Investigated
- While not benchmarked, Radix-8 and split radix is possibly not worth the effort.
  - For radix-4: the number of operations for sample is (3 twiddles complex multiplies (x6) + 8 complex additions (x2) for 4 complex numbers simulating 2 radix-2 stages) = 4.25 real operations per complex sample
  - For radix-8: using special reduction for 1+i: (7 twiddles complex multiplies (x6) + 24 complex additions (x2) + 4 "1+i" multiplication (x2 or x4?) for 8 complex numbers simulating 3 radix-2 stages) =  4.083
  - Theoretically, radix-8 gives a only 4% decrease in time, but this reduction might be less with we consider that the more operations can more easily overload registers.
  - Radix-8 is more complicated.
  - Radix-8 can save time over a Radix-4 + radix 2 stage, but choosing the next radix gets more complex, plus there is only 1 or 0 radix-stages, so speed up will be limited over the entire algorithm
  - Radix-8 could possibly mean less passes over data, maybe increaing performance further with large FFTs.
  - When considering FMA having the same throughput as a simgle Add (or multiply) operation, the the cost of a complex multiplication goes from x6 to x4: which makes radix-8 give NO speedup at all.
  - At max, Split-radix with only 4 real operations per complex sample gives a 5.9% max reduction over radix 4, but is complicated to implement.
  - With FMA even radix-2 have attractive complexity.
- Manually Unrolling the main radix-4 and radix-2 loops does not give much speed up.
- Compiling with Clang gives 5% to 20% speed up over compiling with MSVC or GCC (on Windows, Intel i7-12700)
- Skipping Bit-reversal in convolution is more performant
- Theoretical, Stockham method means no bit-reversal, but adds to each stage of the algorithm. Right now, it is not worth changing the entire algorithm to find out. Additionally, Stockham requires multiple variations SIMD interweave operations, (e.g. interweave every other sample, every other two samples, etc....). This could potentially make relying  Stockham less portable, or more complicated.
-  Index arithmetic with SIMD instructions is NOT performative
  
## Investigating
- Single-pass Bitreversal (Small size)
- SSE Bit reversal (Small Size)
- Unrolled single-pass bitreversal (small size)
- Small cache-oblivious order of bit-reversal "For portability"
- Cache-oblivious order of bit-reversal reorder (medium and large size, compare to COBRA)
- Breaking out Bit-reversal algorithm as a template parameter to the FFT
- In-place operation of main radix-4 and radix-2 loops
- According to Ryg's blog, use FMA more efficient for radix-2 (and maybe radix-4)
- Recursive, Cache-oblivious FFTs
- Main FFT stages performed in-place
- Six stage FFT with 1 vs 2 transposes 

## Inspiration and lessons
- [Python Prototype](https://github.com/goldenrockefeller/fft-prototype)
  - Personal python project for putting the lessons together and testing quickly for correctness
- [PFFFT](https://bitbucket.org/jpommier/pffft/src/master/) and [Github Repo](https://github.com/marton78/pffft)
  - My initial understand of how fast FFTs work
  - Skip bit reversal reordering stage for convolution
  - Align data for faster SIMD computation
  - Most likely using Cooley-Tukey algorithm
- [PGFFT](https://www.shoup.net/PGFFT/)
  - Use COBRA for fast Bit-reversal for large FFT sizes.
  - Consider potential efficiencies for bit-reversal
- [OTFFT](http://wwwa.pikara.ne.jp/okojisan/otfft-en/index.html) and [Github Repo](https://github.com/DEWETRON/otfft)
  - Stockham and Six-stage algorithms instead of Cooley Tukey
  - The fastest of the open-source liberal license FFTs for smaller (<4096 samples) FFTs
- [KFR](https://github.com/kfrlib/fft)
  - Another fast FFT, most likely using Cooley-Tukey
  - Not Liberal License
  - Using Clang to compile can lead to faster performance
- [Ryg's Blog on FFT impentation](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
  - Use DIT and DIF to skip bit reversal reordering stage
  - Radix-4 fft is probaby good enough, considering register usage and code complexity
  - Use FMA more efficient for radix-2
- [Robin Scheibler Blog](http://www.robinscheibler.org/2013/02/13/real-fft.html)
  - Getting Real FFT from Complex FFT
- [Rick Lyons' blog](https://www.dsprelated.com/showarticle/800.php)
  - Using FFT algorithm to implement inverse FFT through a pointer switch (Method #3)
- [MULTIPLY-ADD OPTIMIZED FFT KERNELS] https://www.auer.net/mao.pdf
  - FMA optimized Radix-2  has the lowest complexity! 40% reduction of normal radix-2
  - FMA optimized Radix-4 has the same low complexity as Radix-2
- [Ipp](https://www.intel.com/content/www/us/en/developer/articles/training/how-to-use-intel-ipp-s-1d-fourier-transform-functions.html)
  - A gold standard for FFT, used to verify how efficient AFFT is 


