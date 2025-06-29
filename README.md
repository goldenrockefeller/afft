# afft
Adequately Fast Fourier Transform

This is supposed to be a small C++ project to make a fast, template-based implementaiton of mixed radix (2 and 4) FFT (Fast Fourier Transform), mainly for audio dsp. AFFT will have the ability to be extended with SIMD instructions like SSE and AVX for even more speed.  The goal is to use this code in future audio dsp project. 

Prototype

## Goals
- C++11
- Highly portable, yet efficient
- Template-based for easy adaptation to other platforms and future-proofing (extends to AVX and Neon)
- Header-only
- Powers of 2 (targeting under 2^22)
- Cache-Oblivious where possible
- Liberal license

## Implementation
- Expects different arrays for real and imaginary numbers (for now) (not directly supporting interweaved input, not direction support array or complex numbers)
- Stockham for early stages, Cooley-Tukey (mixed radix of 2 and 4) for later stages
- Use "fast-math" to generate FMA instructions
- Fast Cache-Oblivious SIMD-enabled bit reversal permutation
- Decimation in Time
  
## Investigated
- While not benchmarked, Radix-8 and split radix is possibly not worth the effort.
  - For radix-4: the number of operations for sample is (3 twiddles complex multiplies (x6) + 8 complex additions (x2) for 4 complex numbers simulating 2 radix-2 stages) = 4.25 real operations per complex sample
  - For radix-8: using special reduction for 1+i: (7 twiddles complex multiplies (x6) + 24 complex additions (x2) + 4 "1+i" multiplication (x2 or x4?) for 8 complex numbers simulating 3 radix-2 stages) =  4.083
  - Theoretically, radix-8 gives a only 4% decrease in time, but this reduction might be less with we consider that the more operations can more easily overload registers.
  - Radix-8 is more complicated.
  - Radix-8 can save time over a Radix-4 + radix 2 stage, but choosing the next radix gets more complex, plus there is only 1 or 0 radix-stages, so speed up will be limited over the entire algorithm
  - Radix-8 could possibly mean less passes over data, maybe increaing performance further with large FFTs.
  - At max, Split-radix with only 4 real operations per complex sample gives a 5.9% max reduction over radix 4, but is complicated to implement.
  - With FMA even radix-2 have attractive computational complexity.
- Manually Unrolling the main radix-4 and radix-2 loops does not give much speed up.
- Compiling with Clang gives 5% to 20% speed up over compiling with MSVC or GCC (on Windows, Intel i7-12700)
- Skipping Bit-reversal in convolution is more performant when not using SIMD, but trickier when using SIMD without deinterleave instructions. Estimated savings of 15% - 30%. 
-  Index arithmetic with SIMD instructions is NOT performative
-  Single-pass bitreversal is the fastest. I should aim to minimize the ratio of loads and stores to actual computation and avoid using work pointer
-  Manual loop unrolling doesn't always speed up code.
-  Recursive, Cache-oblivious FFTs works better than iterative at larger sizes.
-  Six-stage or Four-stage fft may increase performance for n_samples > 2^16. Since I am not looking at very large samples sizes for real-time audio processing, I doubt further investigation into this will be worth it.
-  Cache-oblivious bit reversal permutation is significantly faster than my previous COBRA implementation on <2^22. After that, COBRA is slightly faster. This might be because the data no longer fits in my L3 cache and COBRA has a more regular access pattern, and/or also doesn't need to load a bit reversal permutation plan.
-  Offsetting the input real, input imag, output real and output imag arrays by different amounts prevents them from overlapping and overloading the cache-associativity. 5 to 20% bonus when data does not fit in L1.

## Not Investigated
-  It is possible to operate on complex numbers arrays by interleaving and deinterleaving in the first and last steps, respectively. This will also reduce Set-associativity conflicts (as opposed to the offset method mentioned earlier, potentially leading to similar performance gains)
-  The plan following has a small overhead (for loop containing a switch statement). It is possible to remove this overhead for small ffts by pregenerating/hardcoding the end-to-end FFT plan for each length. I will not be doing that at this time.
-  It is possible that hard codinng the twiddle values (like with OTFFT) could reduce loading into data cache and lead to an increase in performance.
-  Radix-8 or Split radix might squeeze out more performance
  

## Inspiration and lessons
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
- [bit-reversed permutation](https://arxiv.org/pdf/1708.01873)
  - Cache-oblivious "recursive" bit-reversed permutation can be as fast or faster than a well-tuned COBRA algorithm
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
- [MULTIPLY-ADD OPTIMIZED FFT KERNELS](https://www.auer.net/mao.pdf)
  - FMA optimized Radix-2  has the lowest complexity! 40% reduction of normal radix-2
  - FMA optimized Radix-4 has the same low complexity as Radix-2
- [Ipp](https://www.intel.com/content/www/us/en/developer/articles/training/how-to-use-intel-ipp-s-1d-fourier-transform-functions.html)
  - A gold standard for FFT, used to verify how efficient AFFT is 


