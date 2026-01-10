#include <iostream>
// #include "afft/complex_fft.hpp"
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
#include "afft/complex_fft.hpp"
#include "afft/real_fft.hpp"
#include "afft/co_real_conv.hpp"
#include "afft/co_real_conv_cached.hpp"
#include "afft/streaming_real_conv.hpp"
#include "afft/spec/val_array_spec.hpp"
#include "afft/spec/double4_avx2_spec.hpp"
#include "afft/spec/double2_sse2_spec.hpp"
#include <random>
#include <iostream>
#include <cmath>

using namespace afft;
using namespace afft::common_math;
using namespace afft::bit_reverse_permute;
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

        ComplexFft<ValArraySpec<OperandSize>> fft(n_samples);

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

template <std::size_t OperandSize>
void check_fft_normalized()
{
    cout << "check_fft_normalized OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        ComplexFft<ValArraySpec<OperandSize>> fft(n_samples);

        auto rng = RandomGenerator();

        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        for (size_t i = 0; i < n_samples; i++)
        {
            pDst[i].re /= static_cast<double>(n_samples);
            pDst[i].im /= static_cast<double>(n_samples);
        }
        fft.fft_normalized(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - y_real[i]) * (pDst[i].re - y_real[i]);
            noise_power_ += (pDst[i].im - y_imag[i]) * (pDst[i].im - y_imag[i]);
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_fft_norm OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

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

template <std::size_t OperandSize>
void check_ifft_normalized()
{
    cout << "check_ifft_normalized OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        ComplexFft<ValArraySpec<OperandSize>> fft(n_samples);

        auto rng = RandomGenerator();

        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        ippsFFTInv_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        for (size_t i = 0; i < n_samples; i++)
        {
            pDst[i].re /= static_cast<double>(n_samples);
            pDst[i].im /= static_cast<double>(n_samples);
        }
        fft.ifft_normalized(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - y_real[i]) * (pDst[i].re - y_real[i]);
            noise_power_ += (pDst[i].im - y_imag[i]) * (pDst[i].im - y_imag[i]);
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_ifft_norm OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

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

template <std::size_t OperandSize>
void check_ifft()
{
    cout << "check_ifft OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        ComplexFft<ValArraySpec<OperandSize>> fft(n_samples);

        auto rng = RandomGenerator();

        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            x_real[i] = r;
            pSrc[i].re = r;

            r = rng.gen();
            x_imag[i] = r;
            pSrc[i].im = r;
        }

        ippsFFTInv_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.ifft(y_real.data(), y_imag.data(), x_real.data(), x_imag.data());

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - y_real[i]) * (pDst[i].re - y_real[i]);
            noise_power_ += (pDst[i].im - y_imag[i]) * (pDst[i].im - y_imag[i]);
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_ifft OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

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
    for (std::size_t i = 1; i < 8; i++)
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

        ComplexFft<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> fft(n_samples);

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

        ComplexFft<Double2Sse2Spec, xsimd::aligned_allocator<double, 1024>> fft(n_samples);

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

template <std::size_t OperandSize>
void check_real_fft()
{
    cout << "check_real_fft OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }

    for (auto n_samples : trials)
    {
        const int order = int_log_2(n_samples);
        IppsFFTSpec_C_64fc *pFFTSpec = 0;
        Ipp8u *pFFTSpecBuf = nullptr, *pFFTInitBuf = nullptr, *pFFTWorkBuf = nullptr;

        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> signal(n_samples);

        int sizeFFTSpec, sizeFFTInitBuf, sizeFFTWorkBuf;
        ippsFFTGetSize_C_64fc(
            order, IPP_FFT_NODIV_BY_ANY,
            ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

        pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
        pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
        pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

        ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
                           ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
        if (pFFTInitBuf)
            ippFree(pFFTInitBuf);

        RealFft<ValArraySpec<OperandSize>> real_fft(n_samples);
        const std::size_t unpacked_len = (n_samples >> 1) + 1;
        std::vector<double> spectra_real(unpacked_len);
        std::vector<double> spectra_imag(unpacked_len);

        auto rng = RandomGenerator();
        for (size_t i = 0; i < n_samples; i++)
        {
            auto r = rng.gen();
            signal[i] = r;
            pSrc[i].re = r;
            pSrc[i].im = 0.0;
        }

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        real_fft.fft(spectra_real.data(), spectra_imag.data(), signal.data());

        
        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples/2 + 1; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - spectra_real[i]) * (pDst[i].re - spectra_real[i]);
            noise_power_ += (pDst[i].im - spectra_imag[i]) * (pDst[i].im - spectra_imag[i]);
            // 
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report
        signal_power_ = 0.0;
        noise_power_ = 0.0;

        if (snr < 200)
        {
            cout << "check_real_fft + nyquist OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

        for (size_t i = 0; i < n_samples/2; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - spectra_real[i]) * (pDst[i].re - spectra_real[i]);
            noise_power_ += (pDst[i].im - spectra_imag[i]) * (pDst[i].im - spectra_imag[i]);
            // 
        }

        snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_real_fft OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr
                 << endl;

            cout << "------------------------------------------- " << endl;
        }

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

template <std::size_t OperandSize>
void check_real_ifft()
{
    cout << "check_real_ifft OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }

    for (auto n_samples : trials)
    {
        RealFft<ValArraySpec<OperandSize>> real_fft(n_samples);
        const std::size_t unpacked_len = (n_samples >> 1) + 1;
        std::vector<double> spectra_real(unpacked_len);
        std::vector<double> spectra_imag(unpacked_len);
        std::vector<double> original(n_samples);
        std::vector<double> reconstructed(n_samples);

        auto rng = RandomGenerator();
        for (size_t i = 0; i < n_samples; i++)
        {
            original[i] = rng.gen();
        }

        real_fft.fft(spectra_real.data(), spectra_imag.data(), original.data());
        real_fft.ifft(reconstructed.data(), spectra_real.data(), spectra_imag.data());

        const double inv_n = 1.0 / static_cast<double>(n_samples);
        for (auto &value : reconstructed)
        {
            value *= inv_n;
        }

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            const double ref = original[i];
            signal_power_ += ref * ref;
            const double diff = ref - reconstructed[i];
            noise_power_ += diff * diff;
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_real_ifft OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr << endl;
            cout << "------------------------------------------- " << endl;
        }
    }
}


template <std::size_t OperandSize>
void check_real_conv()
{
    cout << "check_real_conv OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 13; i++)
    {
        trials.push_back(1 << i);
    }

    for (auto n_samples : trials)
    {
        RealFft<ValArraySpec<OperandSize>> real_fft(n_samples);
        std::vector<double> signal_a(n_samples);
        std::vector<double> signal_b(n_samples);
        std::vector<double> conv_out(n_samples);
        std::vector<double> conv_ref(n_samples);

        auto rng = RandomGenerator();
        for (std::size_t i = 0; i < n_samples; i++)
        {
            signal_a[i] = rng.gen();
            signal_b[i] = rng.gen();
        }

        real_fft.conv(conv_out.data(), signal_a.data(), signal_b.data());

        for (std::size_t i = 0; i < n_samples; i++)
        {
            double acc = 0.0;
            for (std::size_t j = 0; j < n_samples; j++)
            {
                const std::size_t k = (i + n_samples - j) % n_samples;
                acc += signal_a[j] * signal_b[k];
            }
            conv_ref[i] = acc;
        }

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (std::size_t i = 0; i < n_samples; i++)
        {
            const double ref = conv_ref[i];
            signal_power_ += ref * ref;
            const double diff = ref - conv_out[i];
            noise_power_ += diff * diff;
        }

        const double snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_real_conv OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr << endl;
            cout << "------------------------------------------- " << endl;
        }
    }
}

template <std::size_t OperandSize>
void check_real_conv_cached_fft()
{
    cout << "check_real_conv_cached_fft OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 13; i++)
    {
        trials.push_back(1 << i);
    }

    for (auto n_samples : trials)
    {
        RealFft<ValArraySpec<OperandSize>> real_fft(n_samples);
        std::vector<double> signal(n_samples);
        std::vector<double> impulse(n_samples);
        std::vector<double> conv_out(n_samples);
        std::vector<double> conv_ref(n_samples);

        auto rng = RandomGenerator();
        for (std::size_t i = 0; i < n_samples; i++)
        {
            signal[i] = rng.gen();
            impulse[i] = rng.gen();
        }

        std::vector<double> cached_real(real_fft.spectra_len() + 1);
        std::vector<double> cached_imag(real_fft.spectra_len() + 1);
        real_fft.fft(cached_real.data(), cached_imag.data(), impulse.data());
        cached_imag[0] = cached_real[real_fft.spectra_len()];

        real_fft.conv_with_cached_fft(
            conv_out.data(),
            signal.data(),
            cached_real.data(),
            cached_imag.data());

        for (std::size_t i = 0; i < n_samples; i++)
        {
            double acc = 0.0;
            for (std::size_t j = 0; j < n_samples; j++)
            {
                const std::size_t k = (i + n_samples - j) % n_samples;
                acc += signal[j] * impulse[k];
            }
            conv_ref[i] = acc;
        }

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (std::size_t i = 0; i < n_samples; i++)
        {
            const double ref = conv_ref[i];
            signal_power_ += ref * ref;
            const double diff = ref - conv_out[i];
            noise_power_ += diff * diff;
        }

        const double snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_real_conv_cached_fft OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr << endl;
            cout << "------------------------------------------- " << endl;
        }
    }
}

template <std::size_t OperandSize>
void check_co_real_conv()
{
    cout << "check_co_real_conv OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 13; i++)
    {
        trials.push_back(1 << i);
    }

    std::mt19937 chunk_rng(std::random_device{}());

    for (auto n_samples : trials)
    {
        CoRealConv<ValArraySpec<OperandSize>> co_conv(n_samples);

        std::vector<double> signal_a(n_samples);
        std::vector<double> signal_b(n_samples);
        std::vector<double> conv_out(n_samples);
        std::vector<double> conv_ref(n_samples);

        auto rng = RandomGenerator();
        for (std::size_t i = 0; i < n_samples; i++)
        {
            signal_a[i] = rng.gen();
            signal_b[i] = rng.gen();
        }

        const std::size_t total_steps = co_conv.total_steps();
        std::uniform_int_distribution<std::size_t> chunk_dist(1, std::max<std::size_t>(std::size_t(1), total_steps));

        while (!co_conv.finished())
        {
            const std::size_t remaining = co_conv.steps_remaining();
            const std::size_t request = std::min(remaining, chunk_dist(chunk_rng));
            const std::size_t processed = co_conv.process(
                conv_out.data(),
                signal_a.data(),
                signal_b.data(),
                request == 0 ? 1 : request);

            if (processed == 0)
            {
                throw std::runtime_error("CoRealConv stalled during process");
            }
        }

        for (std::size_t i = 0; i < n_samples; i++)
        {
            double acc = 0.0;
            for (std::size_t j = 0; j < n_samples; j++)
            {
                const std::size_t k = (i + n_samples - j) % n_samples;
                acc += signal_a[j] * signal_b[k];
            }
            conv_ref[i] = acc;
        }

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (std::size_t i = 0; i < n_samples; i++)
        {
            const double ref = conv_ref[i];
            signal_power_ += ref * ref;
            const double diff = ref - conv_out[i];
            noise_power_ += diff * diff;
        }

        const double snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_co_real_conv OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr << endl;
            cout << "------------------------------------------- " << endl;
        }
    }
}

template <std::size_t OperandSize>
void check_co_real_conv_cached_fft()
{
    cout << "check_co_real_conv_cached_fft OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 13; i++)
    {
        trials.push_back(1 << i);
    }

    std::mt19937 chunk_rng(std::random_device{}());

    for (auto n_samples : trials)
    {
        if (n_samples == 2)
        {
            continue;
        }

        CoRealConvCached<ValArraySpec<OperandSize>> co_conv(n_samples);

        RealFft<ValArraySpec<OperandSize>> real_fft(n_samples);
        std::vector<double> signal(n_samples);
        std::vector<double> impulse(n_samples);
        std::vector<double> conv_out(n_samples);
        std::vector<double> conv_ref(n_samples);

        auto rng = RandomGenerator();
        for (std::size_t i = 0; i < n_samples; i++)
        {
            signal[i] = rng.gen();
            impulse[i] = rng.gen();
        }

        std::vector<double> cached_real(real_fft.spectra_len() + 1);
        std::vector<double> cached_imag(real_fft.spectra_len() + 1);
        real_fft.fft(cached_real.data(), cached_imag.data(), impulse.data());
        cached_imag[0] = cached_real[real_fft.spectra_len()];

        const std::size_t total_steps = co_conv.total_steps();
        std::uniform_int_distribution<std::size_t> chunk_dist(1, std::max<std::size_t>(std::size_t(1), total_steps));

        while (!co_conv.finished())
        {
            const std::size_t remaining = co_conv.steps_remaining();
            const std::size_t request = std::min(remaining, chunk_dist(chunk_rng));
            const std::size_t processed = co_conv.process(
                conv_out.data(),
                signal.data(),
                cached_real.data(),
                cached_imag.data(),
                request == 0 ? 1 : request);

            if (processed == 0)
            {
                throw std::runtime_error("CoRealConv cached mode stalled during process");
            }
        }

        for (std::size_t i = 0; i < n_samples; i++)
        {
            double acc = 0.0;
            for (std::size_t j = 0; j < n_samples; j++)
            {
                const std::size_t k = (i + n_samples - j) % n_samples;
                acc += signal[j] * impulse[k];
            }
            conv_ref[i] = acc;
        }

        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (std::size_t i = 0; i < n_samples; i++)
        {
            const double ref = conv_ref[i];
            signal_power_ += ref * ref;
            const double diff = ref - conv_out[i];
            noise_power_ += diff * diff;
        }

        const double snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        if (snr < 200)
        {
            cout << "check_co_real_conv_cached_fft OperandSize: " << OperandSize
                 << " N_samples: " << n_samples
                 << " snr: " << snr << endl;
            cout << "------------------------------------------- " << endl;
        }
    }
}

template <std::size_t OperandSize>
void check_streaming_real_conv()
{
    cout << "check_streaming_real_conv OperandSize: " << OperandSize << endl;

    const std::size_t signal_len = 1024;
    const std::size_t impulse_len = 32;

    std::vector<double> signal(signal_len);
    std::vector<double> impulse(impulse_len);

    auto rng = RandomGenerator();
    for (auto &sample : signal)
    {
        sample = rng.gen();
    }
    for (auto &tap : impulse)
    {
        tap = rng.gen();
    }

    StreamingRealConv<ValArraySpec<OperandSize>> streaming_conv(impulse.data(), impulse_len);

    std::vector<double> streaming_output;
    streaming_output.reserve(signal_len);

    std::mt19937 chunk_rng(std::random_device{}());
    std::uniform_int_distribution<std::size_t> chunk_dist_small(1, 7);
    std::bernoulli_distribution chunk_large_prob(0.1);
    std::vector<double> chunk_buffer;

    std::size_t processed = 0;
    while (processed < signal_len)
    {
        std::size_t chunk = chunk_large_prob(chunk_rng) ? 32 : chunk_dist_small(chunk_rng);
        chunk = std::min(signal_len - processed, chunk);
        chunk_buffer.resize(chunk);
        streaming_conv.process_slice(signal.data() + processed, chunk, chunk_buffer.data());
        streaming_output.insert(streaming_output.end(), chunk_buffer.begin(), chunk_buffer.end());
        processed += chunk;
    }

    std::vector<double> reference(signal_len + impulse_len - 1, 0.0);
    for (std::size_t i = 0; i < signal_len; ++i)
    {
        const double x = signal[i];
        for (std::size_t j = 0; j < impulse_len; ++j)
        {
            reference[i + j] += x * impulse[j];
        }
    }

    double signal_power_ = 0.0;
    double noise_power_ = 0.0;
    double max_error = 0.0;

    for (std::size_t i = 0; i < streaming_output.size(); ++i)
    {
        const double ref = reference[i];
        signal_power_ += ref * ref;
        const double diff = ref - streaming_output[i];
        noise_power_ += diff * diff;
        max_error = std::max(max_error, std::abs(diff));
    }

    const double snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);
    const double max_tolerance = 1e-6;
    const bool max_error_ok = max_error <= max_tolerance;

    if (snr < 200 || !max_error_ok)
    {
        cout << "check_streaming_real_conv OperandSize: " << OperandSize
             << " signal_len: " << signal_len
             << " impulse_len: " << impulse_len
             << " snr: " << snr
             << " max_error: " << max_error
             << " tolerance: " << max_tolerance
             << endl;
        cout << "------------------------------------------- " << endl;
    }
}

void do_bench()
{
    cout << "do_bench: " << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 10; i++)
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

        ComplexFft<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> simd_fft(n_samples);

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
            ComplexFft<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> simd_fft_sqr(1 << (int_log_2(n_samples) / 2));
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
                // IppStatus status = ippiTranspose_32fc_C1R(
                //     (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // source pointer and step (row stride in bytes)
                //     (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // destination pointer and step
                //     { (int) sqrt_n, (int)  sqrt_n }                            // size of the source matrix
                // );
                // status = ippiTranspose_32fc_C1R(
                //     (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // source pointer and step (row stride in bytes)
                //     (Ipp32fc*) y_imag, sqrt_n * sizeof(Ipp32fc),        // destination pointer and step
                //     { (int)  sqrt_n,  (int) sqrt_n }                            // size of the source matrix
                // );          
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
    
    
    //do_bench();

    // check_fft<1>();
    // check_fft<2>();
    // check_fft<4>();
    // check_fft<8>();
    // check_fft<16>();
    // check_fft<32>();
    // check_fft<64>();
    // check_fft<128>();
    // check_fft<256>();

    // check_ifft<1>();
    // check_ifft<2>();
    // check_ifft<4>();
    // check_ifft<8>();
    // check_ifft<16>();
    // check_ifft<32>();
    // check_ifft<64>();
    // check_ifft<128>();
    // check_ifft<256>();

    // check_fft_normalized<1>();
    // check_fft_normalized<2>();
    // check_fft_normalized<4>();
    // check_fft_normalized<8>();
    // check_fft_normalized<16>();
    // check_fft_normalized<32>();
    // check_fft_normalized<64>();
    // check_fft_normalized<128>();
    // check_fft_normalized<256>();

    // check_ifft_normalized<1>();
    // check_ifft_normalized<2>();
    // check_ifft_normalized<4>();
    // check_ifft_normalized<8>();
    // check_ifft_normalized<16>();
    // check_ifft_normalized<32>();
    // check_ifft_normalized<64>();
    // check_ifft_normalized<128>();
    // check_ifft_normalized<256>();

    // check_fft_double2sse();
    // check_fft_double4avx();

    check_real_fft<1>();
    check_real_fft<2>();
    check_real_fft<4>();

    check_real_ifft<1>();
    check_real_ifft<2>();
    check_real_ifft<4>();

    check_real_conv<1>();
    check_real_conv<2>();
    check_real_conv<4>();

    check_real_conv_cached_fft<1>();
    check_real_conv_cached_fft<2>();
    check_real_conv_cached_fft<4>();

    check_co_real_conv<1>();
    check_co_real_conv<2>();
    check_co_real_conv<4>();

    check_co_real_conv_cached_fft<1>();
    check_co_real_conv_cached_fft<2>();
    check_co_real_conv_cached_fft<4>();

    check_streaming_real_conv<1>();
    check_streaming_real_conv<2>();
    check_streaming_real_conv<4>();

    return 0;
}

