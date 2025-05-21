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
#include "afft/bit_reverse_permutation/bit_rev_perm_impl.hpp"
#include "afft/bit_reverse_permutation/bit_rev_perm.hpp"
#include "afft/spec/double4_avx2_spec.hpp"
#include "afft/common_math.hpp"
#include "afft/bit_reversal_primitives.hpp"
#include "afft/spec/val_array_spec.hpp"
#include "afft/fft/fft_complex.hpp"
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
void check_bit_rev_perm()
{
    cout << "check_bit_rev_perm OperandSize: " << OperandSize << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_indexes : trials)
    {

        BitRevPerm<StdSpec<double>> bit_rev_perm_double(n_indexes);
        BitRevPerm<ValArraySpec<OperandSize>> bit_rev_perm_valarray(n_indexes);

        std::vector<double, xsimd::aligned_allocator<double, 128>> real_vals(n_indexes);
        std::vector<double, xsimd::aligned_allocator<double, 128>> imag_vals(n_indexes);

        for (std::size_t i = 0; i < n_indexes; i++)
        {
            real_vals[i] = double(i);
            imag_vals[i] = -double(i);
        }

        auto real_vals_double = real_vals;
        auto imag_vals_double = imag_vals;
        auto real_vals_valarray = real_vals;
        auto imag_vals_valarray = imag_vals;

        bit_rev_perm_double.eval(real_vals_double.data(), imag_vals_double.data(), real_vals_double.data(), imag_vals_double.data());
        bit_rev_perm_valarray.eval(real_vals_valarray.data(), imag_vals_valarray.data(), real_vals_valarray.data(), imag_vals_valarray.data());

        bool success = true;
        for (std::size_t i = 0; i < n_indexes; i++)
        {
            success = success && (real_vals_double[i] == real_vals_valarray[i]);
            success = success && (imag_vals_double[i] == imag_vals_valarray[i]);
        }
        if (!success)
        {
            cout << "Failure on Trial: " << n_indexes << ";  Arr_size: " << OperandSize << endl;
        }
    }
}

template <std::size_t OperandSize>
void check_fft_ditime()
{
    cout << "check_fft_ditime OperandSize: " << OperandSize << endl;
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
        std::vector<double> x_real(n_samples);
        std::vector<double> x_imag(n_samples);

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

        FftComplex<ValArraySpec<OperandSize>> fft(n_samples, 1024);

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
        // fft.eval(x_real.data(), x_imag.data(), x_real.data(), x_imag.data());
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     std::cout << "x_real: " << x_real[i] <<endl;
        // }

        /////////////////// COMPUTE

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.eval_ditime(x_real.data(), x_imag.data(), x_real.data(), x_imag.data());

        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - x_real[i]) * (pDst[i].re - x_real[i]);
            noise_power_ += (pDst[i].im - x_imag[i]) * (pDst[i].im - x_imag[i]);
            // std::cout << pDst[i].re << " , " << pDst[i].im << " ; " << x_real[i] << " , " << x_imag[i] <<endl;
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_fft_ditime OperandSize: " << OperandSize
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
void check_fft_difreq()
{
    cout << "check_fft_difreq OperandSize: " << OperandSize << endl;
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
        std::vector<double> x_real(n_samples);
        std::vector<double> x_imag(n_samples);

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

        FftComplex<ValArraySpec<OperandSize>> fft(n_samples, 1024);

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
        // x_imag[2] = 1;
        // fft.eval_difreq(x_real.data(), x_imag.data(), x_real.data(), x_imag.data());
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     std::cout << "x_real, x_imag: " << x_real[i] << " , " << x_imag[i] <<endl;
        // }

        /////////////////// COMPUTE

        ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.eval_difreq(x_real.data(), x_imag.data(), x_real.data(), x_imag.data());

        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - x_real[i]) * (pDst[i].re - x_real[i]);
            noise_power_ += (pDst[i].im - x_imag[i]) * (pDst[i].im - x_imag[i]);
            // std::cout << pDst[i].re << " , " << pDst[i].im << " ; " << x_real[i] << " , " << x_imag[i] <<endl;
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_fft_difreq OperandSize: " << OperandSize
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
void check_fft_inv()
{
    cout << "check_fft_inv OperandSize: " << OperandSize << endl;
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
        std::vector<double> x_real(n_samples);
        std::vector<double> x_imag(n_samples);

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

        FftComplex<ValArraySpec<OperandSize>> fft(n_samples, 1024);

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
        // x_imag[2] = 1;
        // fft.eval_difreq(x_real.data(), x_imag.data(), x_real.data(), x_imag.data());
        // for (size_t i =0 ; i<n_samples; i ++) {
        //     std::cout << "x_real, x_imag: " << x_real[i] << " , " << x_imag[i] <<endl;
        // }

        /////////////////// COMPUTE

        ippsFFTInv_CToC_64fc(pSrc, pDst, pFFTSpec, pFFTWorkBuf);
        fft.eval_difreq(x_imag.data(), x_real.data(), x_imag.data(), x_real.data());

        /////////////////// COMPARE
        double signal_power_ = 0.0;
        double noise_power_ = 0.0;

        for (size_t i = 0; i < n_samples; i++)
        {
            signal_power_ += pDst[i].re * pDst[i].re;
            signal_power_ += pDst[i].im * pDst[i].im;
            noise_power_ += (pDst[i].re - x_real[i]) * (pDst[i].re - x_real[i]);
            noise_power_ += (pDst[i].im - x_imag[i]) * (pDst[i].im - x_imag[i]);
            // std::cout << pDst[i].re << " , " << pDst[i].im << " ; " << x_real[i] << " , " << x_imag[i] <<endl;
        }

        auto snr = 10 * std::log10(signal_power_ / (noise_power_ + 1e-100) + 1e-100);

        // Report

        if (snr < 200)
        {
            cout << "check_fft_difreq OperandSize: " << OperandSize
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

void do_bench()
{
    cout << "do_bench: " << endl;
    std::vector<std::size_t> trials;
    for (std::size_t i = 1; i < 20; i++)
    {
        trials.push_back(1 << i);
    }
    for (auto n_samples : trials)
    {

        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "Br Size: " << n_samples;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        const int order = int_log_2(n_samples);
        // Spec and working buffers
        IppsFFTSpec_C_64fc *pFFTSpec_Fast = 0;
        IppsFFTSpec_C_64fc *pFFTSpec_Accurate = 0;
        Ipp8u *pFFTSpecBuf_Fast, *pFFTInitBuf_Fast, *pFFTWorkBuf_Fast;
        Ipp8u *pFFTSpecBuf_Accurate, *pFFTInitBuf_Accurate, *pFFTWorkBuf_Accurate;

        // Allocate complex buffers
        Ipp64fc *pSrc = ippsMalloc_64fc(n_samples);
        Ipp64fc *pDst = ippsMalloc_64fc(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> x_imag(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_real(n_samples);
        std::vector<double, xsimd::aligned_allocator<double, 1024>> y_imag(n_samples);

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

        FftComplex<StdSpec<double>, xsimd::aligned_allocator<double, 1024>> fft_recursive(n_samples, 1024);
        FftComplex<StdSpec<double>, xsimd::aligned_allocator<double, 1024>> fft_iterative(n_samples);
        FftComplex<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> simd_fft_recursive(n_samples, 1024);
        FftComplex<Double4Avx2Spec, xsimd::aligned_allocator<double, 1024>> simd_fft_iterative(n_samples);

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

        bench.run("Ipp Fast", [&]()
                  { ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec_Fast, pFFTWorkBuf_Fast); });

        // bench.run("Ipp Accurate", [&]()
        //           { ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec_Accurate, pFFTWorkBuf_Accurate); });

        bench.run("OTFFT", [&]() {
            ot_fft->fwd((OTFFT::complex_t*)pSrc);
        });

        // bench.run("fft_recursive_difreq", [&]()
        //           { fft_recursive.eval_difreq(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        // bench.run("fft_recursive_ditime", [&]()
        //           { fft_recursive.eval_ditime(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        // bench.run("fft_iterative_difreq", [&]()
        //           { fft_iterative.eval_difreq(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        // bench.run("fft_iterative_ditime", [&]()
        //           { fft_iterative.eval_ditime(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        bench.run("simd_fft_recursive_difreq", [&]()
                  { simd_fft_recursive.eval_difreq(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        bench.run("simd_fft_recursive_ditime", [&]()
                  { simd_fft_recursive.eval_ditime(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        bench.run("simd_fft_iterative_difreq", [&]()
                  { simd_fft_iterative.eval_difreq(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

        bench.run("simd_fft_iterative_ditime", [&]()
                  { simd_fft_iterative.eval_ditime(y_real.data(), y_imag.data(), x_real.data(), x_imag.data()); });

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
    constexpr std::size_t transformLen = 1 << 6;

    auto bit_reversed_indexes_ = bit_reversed_indexes(transformLen);

    double x_real[8];
    double x_imag[8];

    for (std::size_t i = 0; i < 8; i++)
    {
        x_real[i] = double(i);
        x_imag[i] = -double(i);
    }

    for (std::size_t i = 0; i < 8; i++)
    {
        std::cout << i << " x_real[i], x_imag[i] =" << x_real[i] << ", " << x_imag[i] << std::endl;
    }

    std::cout << "-----------------" << std::endl;

    Double2Sse2Spec::operand x_real_a_op;
    Double2Sse2Spec::operand x_real_b_op;
    Double2Sse2Spec::operand x_real_c_op;
    Double2Sse2Spec::operand x_real_d_op;
    Double2Sse2Spec::operand x_imag_a_op;
    Double2Sse2Spec::operand x_imag_b_op;
    Double2Sse2Spec::operand x_imag_c_op;
    Double2Sse2Spec::operand x_imag_d_op;

    Double2Sse2Spec::load(x_real_a_op, x_real);
    Double2Sse2Spec::load(x_real_b_op, x_real + 2);
    Double2Sse2Spec::load(x_real_c_op, x_real + 4);
    Double2Sse2Spec::load(x_real_d_op, x_real + 6);
    Double2Sse2Spec::load(x_imag_a_op, x_imag);
    Double2Sse2Spec::load(x_imag_b_op, x_imag + 2);
    Double2Sse2Spec::load(x_imag_c_op, x_imag + 4);
    Double2Sse2Spec::load(x_imag_d_op, x_imag + 6);

    Double2Sse2Spec::interleave4(
        x_real_a_op, x_imag_a_op,
        x_real_b_op, x_imag_b_op,
        x_real_c_op, x_imag_c_op,
        x_real_d_op, x_imag_d_op,
        x_real_a_op, x_imag_a_op,
        x_real_b_op, x_imag_b_op,
        x_real_c_op, x_imag_c_op,
        x_real_d_op, x_imag_d_op);

    Double2Sse2Spec::store(x_real, x_real_a_op);
    Double2Sse2Spec::store(x_real + 2, x_real_b_op);
    Double2Sse2Spec::store(x_real + 4, x_real_c_op);
    Double2Sse2Spec::store(x_real + 6, x_real_d_op);
    Double2Sse2Spec::store(x_imag, x_imag_a_op);
    Double2Sse2Spec::store(x_imag + 2, x_imag_b_op);
    Double2Sse2Spec::store(x_imag + 4, x_imag_c_op);
    Double2Sse2Spec::store(x_imag + 6, x_imag_d_op);

    for (std::size_t i = 0; i < 8; i++)
    {
        std::cout << i << " x_real[i], x_imag[i] =" << x_real[i] << ", " << x_imag[i] << std::endl;
    }

    do_bench();

    // check_bit_rev_perm<1>();
    // check_bit_rev_perm<2>();
    // check_bit_rev_perm<4>();
    // check_bit_rev_perm<8>();
    // check_bit_rev_perm<16>();
    // check_bit_rev_perm<32>();
    // check_bit_rev_perm<64>();
    // check_bit_rev_perm<128>();
    // check_bit_rev_perm<256>();

    // check_fft_ditime<1>();
    // check_fft_ditime<2>();
    // check_fft_ditime<4>();
    // check_fft_ditime<8>();
    // check_fft_ditime<16>();
    // check_fft_ditime<32>();
    // check_fft_ditime<64>();
    // check_fft_ditime<128>();
    // check_fft_ditime<256>();

    // check_fft_difreq<1>();
    // check_fft_difreq<2>();
    // check_fft_difreq<4>();
    // check_fft_difreq<8>();
    // check_fft_difreq<16>();
    // check_fft_difreq<32>();
    // check_fft_difreq<64>();
    // check_fft_difreq<128>();
    // check_fft_difreq<256>();

    // check_fft_inv<1>();

    int N = transformLen;
    const int order = (int)(std::log((double)N) / std::log(2.0));

    {
        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "Br Size: " << transformLen;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
        auto ZZ = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2 * 2);
        auto YY = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);

        //     auto ot_fft = OTFFT::Factory::createComplexFFT(N);
        //     auto plan = get_bit_rev_perm_plan(transformLen, 4);

        //     // bench.run("Kiss", [&]() {
        //     //     kiss_fft( cfg , (kiss_fft_cpx*) x ,  (kiss_fft_cpx*) y );
        //     // });

        bench.run("Standard Reversal", [&]()
                  { standard_bitreversal(XX.data(), XX.data() + transformLen, ZZ.data(), ZZ.data() + transformLen, transformLen, bit_reversed_indexes_.data()); });

        bench.run("interleave_bitreversal_single_pass", [&]()
                  { interleave_bitreversal_single_pass(XX.data(), XX.data() + transformLen, ZZ.data(), ZZ.data() + transformLen, transformLen, bit_reversed_indexes_.data()); });

        std::size_t log_len_ = int_log_2(transformLen);
        std::size_t pgfft_brc_thresh = 15;
        std::size_t pgfft_brc_q = 7;
        auto log_reversal_len_ = log_len_ * std::size_t(log_len_ < pgfft_brc_thresh) + (log_len_ - 2 * pgfft_brc_q) * std::size_t(log_len_ >= pgfft_brc_thresh);

        auto bit_reversed_indexes__ = bit_reversed_indexes(1 << log_reversal_len_);
        auto bit_reversed_indexes_2_ = bit_reversed_indexes(1L << std::min(pgfft_brc_q, log_len_));

        if (log_len_ >= pgfft_brc_thresh)
        {
            bench.run("cobra", [&]()
                      { cobra<Double4Avx2Spec>(XX.data(), XX.data() + transformLen, ZZ.data(), ZZ.data() + transformLen, YY.data(), YY.data() + transformLen, bit_reversed_indexes__, bit_reversed_indexes_2_, log_reversal_len_); });
        }

        BitRevPerm<Double4Avx2Spec> bit_rev_perm(transformLen);

        bench.run("bit_rev_perm", [&]()
                  { bit_rev_perm.eval(XX.data(), XX.data() + transformLen, XX.data(), XX.data() + transformLen); });

        if (transformLen >= 16)
        {
            BitRevPermImpl<Double2Sse2Spec> bit_rev_perm_impl(transformLen);

            bench.run("bit_rev_perm_impl2", [&]()
                      { bit_rev_perm_impl.eval(XX.data(), XX.data() + transformLen, XX.data(), XX.data() + transformLen); });
        }

        BitRevPermImpl<StdSpec<double>> bit_rev_perm_simple(transformLen);

        bench.run("bit_rev_perm_simple", [&]()
                  { bit_rev_perm_simple.eval(XX.data(), XX.data() + transformLen, XX.data(), XX.data() + transformLen); });

        //     auto bit_reversed_indexes_16 = bit_reversed_indexes(16);

        // bench.run("interleave_bitreversal_single_pass_by_16", [&]() {
        //     interleave_bitreversal_single_pass_by_16(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen, bit_reversed_indexes_16.data());
        // });

        std::cout << XX[6] << std::endl;
    }

    // {
    //     ankerl::nanobench::Bench bench;
    //     ostringstream title_stream;
    //     title_stream << "Radix, Size: " << transformLen;
    //     bench.title(title_stream.str());

    //     bench.minEpochIterations(10);

    //     auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
    //     auto ZZ = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
    //     auto YY = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);

    //     bench.run("base_radix_8_fma", [&]() {
    //         base_radix_8_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_8_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //     });

    //     bench.run("base_radix_2_fma", [&]() {
    //         base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //     });

    //     bench.run("base_radix_2", [&]() {
    //         base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //     });

    //     bench.run("base_radix_4_fma", [&]() {
    //         base_radix_4_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_4_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_4_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //     });

    //     bench.run("base_radix_4", [&]() {
    //         base_radix_4<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_4<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //         base_radix_4<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
    //     });

    //     bench.run("do_radix4_ditime_regular_core_stage", [&]() {
    //         do_radix4_ditime_regular_core_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //         do_radix4_ditime_regular_core_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //         do_radix4_ditime_regular_core_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //     });

    //     bench.run("do_radix4_ditime_regular_core_oop_stage", [&]() {
    //         do_radix4_ditime_regular_core_oop_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, YY.data(), YY.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //         do_radix4_ditime_regular_core_oop_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, YY.data(), YY.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //         do_radix4_ditime_regular_core_oop_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, YY.data(), YY.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //     });

    //     bench.run("do_radix4_difreq_regular_core_stage", [&]() {
    //         do_radix4_difreq_regular_core_stage<double, Double4Spec, true>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //         do_radix4_difreq_regular_core_stage<double, Double4Spec, true>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //         do_radix4_difreq_regular_core_stage<double, Double4Spec, true>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
    //     });

    //     std::cout << XX[6] << std::endl;
    // }

    // // Spec and working buffers
    // IppsFFTSpec_C_64fc *pFFTSpec=0;
    // Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

    // // Allocate complex buffers
    // Ipp64fc *pSrc=ippsMalloc_64fc(N);
    // Ipp64fc *pDst=ippsMalloc_64fc(N);

    // // Query to get buffer sizes
    // int sizeFFTSpec,sizeFFTInitBuf,sizeFFTWorkBuf;
    // ippsFFTGetSize_C_64fc(order, IPP_FFT_NODIV_BY_ANY,
    //     ippAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

    // // Alloc FFT buffers
    // pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
    // pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
    // pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

    // // Initialize FFT
    // ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY,
    //     ippAlgHintFast, pFFTSpecBuf, pFFTInitBuf);
    // if (pFFTInitBuf) ippFree(pFFTInitBuf);

    //         ippsFFTFwd_CToC_64fc(pSrc,pDst,pFFTSpec,pFFTWorkBuf);

    // // Do the FFT
    // {
    //     ankerl::nanobench::Bench bench;
    //     ostringstream title_stream;
    //     title_stream << "Size: " << transformLen;
    //     bench.title(title_stream.str());

    //     bench.minEpochIterations(10);

    //     auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
    //     auto ZZ =std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);

    //     auto ot_fft = OTFFT::Factory::createComplexFFT(N);

    //     // bench.run("Kiss", [&]() {
    //     //     kiss_fft( cfg , (kiss_fft_cpx*) x ,  (kiss_fft_cpx*) y );
    //     // });

    //     bench.run("PGFFT", [&]() {
    //         pgfft.apply(x,  y);
    //     });

    //     bench.run("OTFFT", [&]() {
    //         ot_fft->fwd((OTFFT::complex_t*)X);
    //     });

    //     // bench.run("PFFFT", [&]() {
    //     //     pffftd_transform(ffts, (double*) x,  (double*) y, W, PFFFT_FORWARD);
    //     // });

    //     bench.run("AFFT", [&]() {
    //         fft.process_dit(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, false, false, false);
    //     });

    //     bench.run("AFFT Slow", [&]() {
    //         fft_slow.process_dit(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen);
    //     });

    //     bench.run("Ipp", [&]() {
    //         std::memcpy(pSrc, XX.data(), transformLen * 2 * sizeof(double));
    //         ippsFFTFwd_CToC_64fc(pSrc,pDst,pFFTSpec,pFFTWorkBuf);
    //     });
    // }

    // {
    //     ankerl::nanobench::Bench bench;
    //     ostringstream title_stream;
    //     title_stream << "New: " << transformLen;
    //     bench.title(title_stream.str());

    //     bench.minEpochIterations(10);

    //     for (size_t i = 4; i <= 19; i++) {
    //         auto transformLen2 = 1 << i;
    //         auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen2 * 2);
    //         auto ZZ =std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen2 * 2);
    //         FftComplex<StdSpec<double>, Double4Spec> fft2(transformLen2);
    //         ostringstream name;
    //         name << "FFT" << transformLen2;
    //         bench.run(name.str(), [&]() {
    //             fft2.process_dit(XX.data(), XX.data()+transformLen2, ZZ.data(), ZZ.data()+transformLen2, false, false, false);
    //         });
    //     }
    // }

    // //--------------------------------------------------------------------------

    // std::size_t signal_len = 8;
    // std::size_t spectra_len = (signal_len >> 1) + 1 ;
    // FftReal<StdSpec<double>, Double4Spec> fft_real(signal_len);
    // {
    //     std::vector<double> signal({1,22,3,4,1,22,3,4});
    //     std::vector<double> spectra_real(spectra_len);
    //     std::vector<double> spectra_imag(spectra_len);

    //     fft_real.compute_spectra(
    //         spectra_real.data(),
    //         spectra_imag.data(),
    //         signal.data()
    //     );

    //     for (int i = 0; i < spectra_len; i++)
    //     {
    //         std::cout << spectra_real[i] << ", " <<  spectra_imag[i] << std::endl;
    //     }
    // }

    // {
    //     std::vector<double> signal(signal_len);
    //     std::vector<double> spectra_real({61, 0, -5, 0, -43});
    //     std::vector<double> spectra_imag({0, 1, -36, -1, 0});

    //     fft_real.compute_signal(
    //         signal.data(),
    //         spectra_real.data(),
    //         spectra_imag.data()
    //     );

    //     for (int i = 0; i < signal_len; i++)
    //     {
    //         std::cout << signal[i] << std::endl;
    //     }
    // }

    // // -------------------------------------------------------------------------

    // {
    //     std::cout << "Convolution:" << std::endl;
    //     std::size_t signal_len_conv = 8;
    //     ConvolutionReal<StdSpec<double>, Double4Spec> conv(signal_len_conv);
    //     std::vector<double> signal({1,1,1,1,0,0,0,0});
    //     std::vector<double> signal_b({1,1,0,0,0,0,0,0});
    //     std::vector<double> signal_auto_conv(signal_len_conv);
    //     conv.compute_convolution(
    //         signal_auto_conv.data(),
    //         signal.data(),
    //         signal_b.data(),
    //         true
    //     );
    //     for (int i = 0; i < signal_len_conv; i++)
    //     {
    //         std::cout << signal_auto_conv[i] << std::endl;
    //     }
    // }

    // {
    //     ConvolutionReal<StdSpec<double>, Double4Spec> conv(transformLen);
    //     std::vector<double> signal(transformLen);
    //     std::vector<double> signal_b(transformLen);
    //     std::vector<double> signal_auto_conv(transformLen);
    //     std::vector<double> signal_auto_conv_b(transformLen);

    //     std::random_device rd;  // Non-deterministic random number generator
    //     std::mt19937 gen(rd()); // Mersenne Twister engine
    //     std::uniform_real_distribution<> dis(0.0, 1.0);

    //     for (int i = 0; i < signal.size(); ++i) {
    //         signal[i] = dis(gen);
    //         signal_b[i] = dis(gen);
    //     }

    //     ankerl::nanobench::Bench bench;
    //     ostringstream title_stream;
    //     title_stream << "Convolution: " << transformLen;
    //     bench.title(title_stream.str());

    //     bench.minEpochIterations(10);

    //     bench.run("Fast Convol", [&]() {
    //         conv.compute_convolution(
    //             signal_auto_conv.data(),
    //             signal.data(),
    //             signal_b.data(),
    //             true
    //         );
    //     });

    //     bench.run("Slow Convol", [&]() {
    //         conv.compute_convolution(
    //             signal_auto_conv.data(),
    //             signal.data(),
    //             signal_b.data(),
    //             false
    //         );
    //     });

    //     conv.compute_convolution(
    //         signal_auto_conv.data(),
    //         signal.data(),
    //         signal_b.data(),
    //         true
    //     );

    //     conv.compute_convolution(
    //         signal_auto_conv_b.data(),
    //         signal.data(),
    //         signal_b.data(),
    //         false
    //     );

    //     double conv_diff = 0;

    //     for (int i = 0; i < signal.size(); ++i) {
    //         conv_diff += abs(signal_auto_conv[i] - signal_auto_conv_b[i]);
    //     }

    //     cout << "Conv diff: " << conv_diff << endl;
    // }

    // pffftd_aligned_free(W);
    // pffftd_aligned_free(Y);
    // pffftd_aligned_free(X);
    // pffftd_aligned_free(Z);
    // pffftd_destroy_setup(ffts);
    // kiss_fft_free(cfg);

    return 0;
}