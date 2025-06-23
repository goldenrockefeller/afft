#include <iostream>
// #include "afft/fft_complex.hpp"
// #include "afft/fft_real.hpp"
// #include "afft/spec/double4_avx2_spec.hpp"
// #include "afft/convolution_real.hpp"
#include "pffft_double.h"
// #include "fft.h"
#include "PGFFT.h"
#include "kiss_fft.h"
#include "nanobench.h"
#include "ipp.h"
#include <sstream>
#include <random>
#include "otfft.h"
#include "afft/fft_complex.hpp"
#include "afft/spec/val_array_spec.hpp"
#include "afft/spec/double4_avx2_spec.hpp"
#include "afft/spec/double2_sse2_spec.hpp"
#include <random>
#include <iostream>
#include <cmath>

using namespace afft;
using namespace afft::common_math;
using namespace afft::plan_indexes_manipulation;
using namespace std;

class RandomGenerator
{
public:
    RandomGenerator(double min = -1.0, double max = 1.0)
        : rng(std::random_device{}()), dist(min, max) {}

    double gen()
    {
        return dist(rng);
    }

private:
    std::mt19937 rng;                            // Mersenne Twister RNG
    std::uniform_real_distribution<double> dist; // Uniform distribution for doubles
};


template <std::size_t OperandSize>
void check_fft()
{
    cout << "check_fft OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 18; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        // Spec and working buffers
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

        // Allocate complex buffers
        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

        // Query to get buffer sizes
        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        // Alloc FFT buffers
        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        // Initialize FFT
        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        FftComplex<ValArraySpec<OperandSize>> fft(n_samples);

        auto rng = RandomGenerator();

        // Set up random number
        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        /////////////////// DeBUG
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     x_real[i] = 0;
        //     x_imag[i] = 0;
        // }
        // x_real[0] = 1;
        // std::cout << "DEBUG" << std::endl;
        // fft.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());
        // std::cout << "x_real[0] " << x_real[0] << std::endl;
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     std::cout << "y_real: " << y_real[i] <<endl;
        // }
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     std::cout << "y_imag: " << y_imag[i] <<endl;
        // }
        /////////////////// COMPUTE

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());

        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - y_real[i]) * (pDst[i].re - y_real[i]);
            noise_power_ += (pDst[i].im - y_imag[i]) * (pDst[i].im - y_imag[i]);
            // 
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_fft OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

        /////////////////// CLEANUP

        if (pSrc)
            ippFree(pSrc);

        if (pDst)
            ippFree(pDst);

        if (pFFTSpecBuf)
            ippFree(pFFTSpecBuf);

        if (pFFTWorkBuf)
            ippFree(pFFTWorkBuf);
    }
}

void check_fft_double4avx()
{
    cout << "check_fft_double4avx" << 4 << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        // Spec and working buffers
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

        // Allocate complex buffers
        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

        // Query to get buffer sizes
        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        // Alloc FFT buffers
        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        // Initialize FFT
        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        FftComplex<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> fft(n_samples);

        auto rng = RandomGenerator();

        // Set up random number
        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        /////////////////// DeBUG
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     x_real[i] = 0;
        //     x_imag[i] = 0;
        // }
        // x_real[0] = 1;
        // std::cout << "DEBUG" << std::endl;
        // fft.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     std::cout << "x_real: " << x_real[i] <<endl;
        // }

        /////////////////// COMPUTE

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());

        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - y_real[i]) * (pDst[i].re - y_real[i]);
            noise_power_ += (pDst[i].im - y_imag[i]) * (pDst[i].im - y_imag[i]);
            // 
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_fft Avx Double 4: " 
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

        /////////////////// CLEANUP

        if (pSrc)
            ippFree(pSrc);

        if (pDst)
            ippFree(pDst);

        if (pFFTSpecBuf)
            ippFree(pFFTSpecBuf);

        if (pFFTWorkBuf)
            ippFree(pFFTWorkBuf);
    }
}

void check_fft_double2sse()
{
    cout << "check_fft_double2sse" << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 7; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        // Spec and working buffers
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

        // Allocate complex buffers
        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

        // Query to get buffer sizes
        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        // Alloc FFT buffers
        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        // Initialize FFT
        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        FftComplex<Double2Sse2Spec, xsimd::aligned_allocator<double, 1024>> fft(n_samples);

        auto rng = RandomGenerator();

        // Set up random number
        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        /////////////////// DeBUG
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     x_real[i] = 0;
        //     x_imag[i] = 0;
        // }
        // x_imag[0] = 1;
        // std::cout << "DEBUG" << std::endl;
        // fft.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());
        // for (size_t i = 0 ; i<n_samples; i ++) {
        //     std::cout << "y_real: " << y_real[i] <<endl;
        // }
        // for (size_t i = 0 ; i<n_samples; i ++) {
        //     std::cout << "y_imag: " << y_imag[i] <<endl;
        // }

        /////////////////// COMPUTE

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());

        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - y_real[i]) * (pDst[i].re - y_real[i]);
            noise_power_ += (pDst[i].im - y_imag[i]) * (pDst[i].im - y_imag[i]);
            // 
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_fft Sse Double 2: " 
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "-------------------------------------------- " << endl;
        }
 
        /////////////////// CLEANUP

        if (pSrc)
            ippFree(pSrc);

        if (pDst)
            ippFree(pDst);

        if (pFFTSpecBuf)
            ippFree(pFFTSpecBuf);

        if (pFFTWorkBuf)
            ippFree(pFFTWorkBuf);
    }
}

void do_bench()
{
    cout << "do_bench: " << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 19; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {

        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "n_samples: " << n_samples;
        bench.title(title_stream.str());
        bench.relative(true);

        const int order = int_log_2(n_samples);
        // Spec and working buffers
        IppsFFTSpec_C_64fc *pFFTSpec_Fast = 0;
        IppsFFTSpec_C_64fc *pFFTSpec_Accurate = 0;
        Ipp8u *pFFTSpecBuf_Fast, *pFFTInitBuf_Fast, *pFFTWorkBuf_Fast;
        Ipp8u *pFFTSpecBuf_Accurate, *pFFTInitBuf_Accurate, *pFFTWorkBuf_Accurate;

        // Allocate complex buffers
        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> data(4 * n_samples + 256 * 6);
        auto x_real = data.data();
        auto y_real = data.data() + n_samples;
        auto x_imag = data.data() + 2 * n_samples;
        auto y_imag = data.data() + 3 * n_samples;
        auto x_realoff = data.data();
        auto y_realoff = data.data() + n_samples + 256;
        auto x_imagoff = data.data() + 2 * n_samples + 128;
        auto y_imagoff = data.data() + 3 * n_samples + 128 * 3;

        // Query to get buffer sizes
        int sizeFFTSpec_Fast, sizeFFTInitBuf_Fast, sizeFFTWorkBuf_Fast;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec_Fast, &sizeFFTInitBuf_Fast, &sizeFFTWorkBuf_Fast);

        int sizeFFTSpec_Accurate, sizeFFTInitBuf_Accurate, sizeFFTWorkBuf_Accurate;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintAccurate, &sizeFFTSpec_Accurate, &sizeFFTInitBuf_Accurate, &sizeFFTWorkBuf_Accurate);

        // Alloc FFT buffers
        pFFTSpecBuf_Fast = ippsMalloc_8u(sizeFFTSpec_Fast);
        pFFTInitBuf_Fast = ippsMalloc_8u(sizeFFTInitBuf_Fast);
        pFFTWorkBuf_Fast = ippsMalloc_8u(sizeFFTWorkBuf_Fast);

        // Alloc FFT buffers
        pFFTSpecBuf_Accurate = ippsMalloc_8u(sizeFFTSpec_Accurate);
        pFFTInitBuf_Accurate = ippsMalloc_8u(sizeFFTInitBuf_Accurate);
        pFFTWorkBuf_Accurate = ippsMalloc_8u(sizeFFTWorkBuf_Accurate);

        // Initialize FFT
        ippsFFTInit_C_64fc(&pFFTSpec_Fast, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf_Fast, pFFTInitBuf_Fast);
        ippsFFTInit_C_64fc(&pFFTSpec_Accurate, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintAccurate, pFFTSpecBuf_Accurate, pFFTInitBuf_Accurate);
        

        if (pFFTInitBuf_Fast)
            ippFree(pFFTInitBuf_Fast);
        if (pFFTInitBuf_Accurate)
            ippFree(pFFTInitBuf_Accurate);

        
        auto ot_fft = OTFFT::Factory::createComplexFFT(n_samples);

        FftComplex<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> simd_fft(n_samples);

        auto rng = RandomGenerator();

        // Set up random number
        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        /////////////////// COMPUTE
        
        bench.epochIterations(1000);

        bench.run("Ipp Fast", [&]()
                  { ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec_Fast, pFFTWorkBuf_Fast); });

        // bench.run("Ipp Accurate", [&]()
        //           { ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec_Accurate, pFFTWorkBuf_Accurate); });

        bench.run("OTFFT", [&]() {
            ot_fft->fwd((OTFFT::complex_t*)pSrc);
        });

        // bench.run("fft_recursive_difreq", [&]()
        //           { fft_recursive.eval_difreq(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        // bench.run("fft_recursive", [&]()
        //           { fft_recursive.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        // bench.run("fft_iterative_difreq", [&]()
        //           { fft_iterative.eval_difreq(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        // bench.run("fft_iterative", [&]()
        //           { fft_iterative.eval(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });


        bench.run("AFFT", [&]()
                  { simd_fft.eval(y_real, y_imag, x_real, x_imag); });

        bench.run("AFFToff", [&]()
            { simd_fft.eval(y_realoff, y_imagoff, x_realoff, x_imagoff); });



        if (n_samples > 16) {
            
            std::size_t sqrt_n = 1 << (int_log_2(n_samples) / 2);
            FftComplex<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> simd_fft_sqr(1 << (int_log_2(n_samples) / 2));
            bench.run("AFFT 4-step", [&]()
            { 

                for(std::size_t i = 0; i < 2 * (1 + (sqrt_n * sqrt_n < n_samples)); i ++) {
                    IppStatus status = ippiTranspose_32fc_C1R(
                        (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // source pointer and step (row stride in bytes)
                        (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // destination pointer and step
                        { (int) sqrt_n, (int)  sqrt_n }                            // size of the source matrix
                    );
                    status = ippiTranspose_32fc_C1R(
                        (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // source pointer and step (row stride in bytes)
                        (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // destination pointer and step
                        { (int)  sqrt_n,  (int) sqrt_n }                            // size of the source matrix
                    );
                    for(std::size_t j = 0; j < sqrt_n; j ++) {
                        simd_fft_sqr.eval(y_real + j * sqrt_n, y_imag  + j * sqrt_n, x_real + j * sqrt_n, x_imag + j * sqrt_n);
                    }
                }  
                IppStatus status = ippiTranspose_32fc_C1R(
                    (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // source pointer and step (row stride in bytes)
                    (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // destination pointer and step
                    { (int) sqrt_n, (int)  sqrt_n }                            // size of the source matrix
                );
                status = ippiTranspose_32fc_C1R(
                    (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // source pointer and step (row stride in bytes)
                    (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // destination pointer and step
                    { (int)  sqrt_n,  (int) sqrt_n }                            // size of the source matrix
                );          
            });
        }

        std::cout << y_real[0] << pDst[0].re << std::endl;

        /////////////////// CLEANUP

        if (pSrc)
            ippFree(pSrc);

        if (pDst)
            ippFree(pDst);

        if (pFFTSpecBuf_Fast)
            ippFree(pFFTSpecBuf_Fast);

        if (pFFTWorkBuf_Fast)
            ippFree(pFFTWorkBuf_Fast);

        if (pFFTSpecBuf_Accurate)
            ippFree(pFFTSpecBuf_Accurate);

        if (pFFTWorkBuf_Accurate)
            ippFree(pFFTWorkBuf_Accurate);
    }
}

int main()
{
    
    
    do_bench();

    check_fft<1>();
    // check_fft<2>();
    // check_fft<4>();
    // check_fft<8>();
    // check_fft<16>();
    // check_fft<32>();
    // check_fft<64>();
    // check_fft<128>();
    // check_fft<256>();

    check_fft_double2sse();
    check_fft_double4avx();

    
    std::cout << "Done." << std::endl;

    return 0;
}