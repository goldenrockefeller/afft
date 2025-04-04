#include <iostream>
#include "afft/fft_complex.hpp"
#include "afft/fft_real.hpp"
#include "afft/spec.hpp"
#include "afft/convolution_real.hpp"
#include "pffft_double.h"
//#include "fft.h"
#include "PGFFT.h"
#include "kiss_fft.h"
#include "nanobench.h"
#include "ipp.h"
#include <sstream>
#include <random>
#include "otfft.h"

using namespace std;
using namespace afft;

template <typename Sample> 
struct OperandSpec{
    using Value = Sample[1];
};

#include "afft/bit_reversal_prototypes.hpp"
#include "afft/radix_primitives.hpp"

int main() {
    constexpr std::size_t transformLen = 1 << 6;
    // 256, double, double, forward
    // 512, double, double, forward
    // 32, double, AVX, forward
    // 256, double AVX, forward
    // 512, double AVX, forward

    PFFFTD_Setup *ffts = pffftd_new_setup(transformLen, PFFFT_COMPLEX);
    PGFFT pgfft(transformLen);
    kiss_fft_cfg cfg=kiss_fft_alloc(transformLen,0,NULL,NULL);

    auto bit_reversed_indexes_ = bit_reversed_indexes(transformLen);
    
    double *X = (double*)pffftd_aligned_malloc(transformLen * 2 * sizeof(double));  /* complex: re/im interleaved */
    double *Y = (double*)pffftd_aligned_malloc(transformLen * 2 * sizeof(double));  /* complex: re/im interleaved */
    double *Z = (double*)pffftd_aligned_malloc(transformLen * 2 * sizeof(double));  /* complex: re/im not-interleaved */
    double *W = (double*)pffftd_aligned_malloc(transformLen * 2 * sizeof(double));

    // /* prepare some input data */
    // for (int k = 0; k < 2 * transformLen; k += 4)
    // {
    //     X[k] =  k / 2;  /* real */
    //     X[k+1] = (k / 2) & 1;  /* imag */

    //     X[k+2] = -1 - k / 2;  /* real */
    //     X[k+3] = (k / 2) & 1;  /* imag */
    // }
    
    std::complex<double> *x = (std::complex<double> *)X;
    std::complex<double> *y = (std::complex<double> *)Y;

    for (int k = 0; k < transformLen; k += 2)
    {
        x[k] =  std::complex<double>(k/2, (k / 2) & 1);
        x[k+1] = std::complex<double>( -1 - k / 2, (k / 2) & 1);  /* imag */
    }

    // pffftd_transform(ffts, (double*) x,  (double*) y, W, PFFFT_FORWARD);
    //pgfft.apply(x,  y);
    kiss_fft( cfg , (kiss_fft_cpx*) x ,  (kiss_fft_cpx*) y );
    
    FftComplex<StdSpec<double>, Double4Spec> fft(transformLen);
    FftComplex<StdSpec<double>, StdSpec<double>> fft_slow(transformLen);

    ippSetNumThreads(1);

    /* prepare some input data */
    for (int k = 0; k < transformLen; k += 2)
    {
        Z[k] =  k / 2;  /* real */
        Z[k+transformLen] = (k / 2) & 1;  /* imag */

        Z[k+1] = -1 - k / 2;  /* real */
        Z[k+1+transformLen] = (k / 2) & 1;  /* imag */
    }

    fft.process_dit(X, X+transformLen, Z, Z+transformLen);

    /* compare output data */
    double diff = 0;
    for (int k = 0; k < transformLen; k += 1)
    {
        double new_diff = abs(Y[2 * k] - X[k]) + abs(Y[2 * k + 1] - X[k + transformLen]);

        if (new_diff > diff) {
            diff = new_diff;
        }
    }

    cout << diff << endl;
 
    cout << PGFFT::simd_enabled() << endl;

    // std::vector<std::size_t> trials = {16, 32, 64, 128, 256, 512, 1024};
    // for (auto n_indexes : trials){
    //     cout << "Trial: " << n_indexes << endl;
    //     auto plan = get_bit_rev_perm_plan(n_indexes, 4);

    //     std::vector<double, xsimd::aligned_allocator<double, 128>> real_vals(n_indexes);
    //     std::vector<double, xsimd::aligned_allocator<double, 128>> imag_vals(n_indexes);

    //     for (std::size_t i = 0; i < n_indexes; i++) {
    //         real_vals[i] = double(i);
    //         imag_vals[i] = - double(i);
    //     }

    //     cache_oblivious_bit_reversal_permutation<double, Double4Spec>(real_vals.data(), imag_vals.data(), plan);
        
    //     for (const auto& val : real_vals) {
    //         cout << val << " ";
    //     }
    //     cout << endl << "-----------" << endl;
    //     for (const auto& val : imag_vals) {
    //         cout << val << " ";
    //     }
    //     cout << endl << "-----------" << endl;
    // }


    int N = transformLen;
    const int order=(int)(std::log((double)N)/std::log(2.0));

    {
        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "Br Size: " << transformLen;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
        auto ZZ =std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2 *2);
        auto YY =std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);

        auto ot_fft = OTFFT::Factory::createComplexFFT(N);
        auto plan = get_bit_rev_perm_plan(transformLen, 4);

        // bench.run("Kiss", [&]() {
        //     kiss_fft( cfg , (kiss_fft_cpx*) x ,  (kiss_fft_cpx*) y );
        // });

        bench.run("Standard Reversal", [&]() {
            standard_bitreversal(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen, bit_reversed_indexes_.data());
        });

        
        bench.run("interleave_bitreversal_single_pass", [&]() {
            interleave_bitreversal_single_pass(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen, bit_reversed_indexes_.data());
        });

        bench.run("cache_oblivious_bit_reversal_permutation", [&]() {
            cache_oblivious_bit_reversal_permutation<double, Double4Spec>(XX.data(), XX.data()+transformLen, plan);
        });

        std::size_t log_len_ = int_log_2(transformLen);
        std::size_t pgfft_brc_thresh = 15;
        std::size_t pgfft_brc_q = 7;
        auto log_reversal_len_ = log_len_ * std::size_t(log_len_ < pgfft_brc_thresh)
            + (log_len_ - 2 * pgfft_brc_q) 
            * std::size_t(log_len_ >= pgfft_brc_thresh);
               
        auto bit_reversed_indexes__ = bit_reversed_indexes(1 << log_reversal_len_);
        auto bit_reversed_indexes_2_ = bit_reversed_indexes(1L << std::min(pgfft_brc_q, log_len_));

        if (log_len_ >= pgfft_brc_thresh) {
            bench.run("cobra", [&]() {
                cobra<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, YY.data(),  YY.data()+transformLen, bit_reversed_indexes__, bit_reversed_indexes_2_, log_reversal_len_);
            });
        }

        auto bit_reversed_indexes_16 = bit_reversed_indexes(16);

        bench.run("interleave_bitreversal_single_pass_by_16", [&]() {
            interleave_bitreversal_single_pass_by_16(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen, bit_reversed_indexes_16.data());
        });

        std::cout << XX[6] << std::endl;
    
    }

    

    {
        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "Radix, Size: " << transformLen;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
        auto ZZ = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
        auto YY = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);

        bench.run("base_radix_8_fma", [&]() {
            base_radix_8_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_8_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
        });

        bench.run("base_radix_2_fma", [&]() {
            base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
        });

        bench.run("base_radix_2", [&]() {
            base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_2<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
        });

        bench.run("base_radix_4_fma", [&]() {
            base_radix_4_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_4_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_4_fma<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
        });

        bench.run("base_radix_4", [&]() {
            base_radix_4<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_4<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
            base_radix_4<double, Double4Spec>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, transformLen);
        });

        bench.run("do_radix4_ditime_regular_core_stage", [&]() {
            do_radix4_ditime_regular_core_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
            do_radix4_ditime_regular_core_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
            do_radix4_ditime_regular_core_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
        });


        bench.run("do_radix4_ditime_regular_core_oop_stage", [&]() {
            do_radix4_ditime_regular_core_oop_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, YY.data(), YY.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
            do_radix4_ditime_regular_core_oop_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, YY.data(), YY.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
            do_radix4_ditime_regular_core_oop_stage<double, Double4Spec, false>(XX.data(), XX.data()+transformLen, YY.data(), YY.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
        });
        

        bench.run("do_radix4_difreq_regular_core_stage", [&]() {
            do_radix4_difreq_regular_core_stage<double, Double4Spec, true>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
            do_radix4_difreq_regular_core_stage<double, Double4Spec, true>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
            do_radix4_difreq_regular_core_stage<double, Double4Spec, true>(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, 0, transformLen/16, 4, 0, 4);
        });    

        std::cout << XX[6] << std::endl;
    }

    // Spec and working buffers
    IppsFFTSpec_C_64fc *pFFTSpec=0;
    Ipp8u *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;

    // Allocate complex buffers
    Ipp64fc *pSrc=ippsMalloc_64fc(N);
    Ipp64fc *pDst=ippsMalloc_64fc(N); 

    // Query to get buffer sizes
    int sizeFFTSpec,sizeFFTInitBuf,sizeFFTWorkBuf;
    ippsFFTGetSize_C_64fc(order, IPP_FFT_NODIV_BY_ANY, 
        ippAlgHintAccurate, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);

    // Alloc FFT buffers
    pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
    pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
    pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);

    // Initialize FFT
    ippsFFTInit_C_64fc(&pFFTSpec, order, IPP_FFT_NODIV_BY_ANY, 
        ippAlgHintAccurate, pFFTSpecBuf, pFFTInitBuf);
    if (pFFTInitBuf) ippFree(pFFTInitBuf);

    // Do the FFT
    {
        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "Size: " << transformLen;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);
        auto ZZ =std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen * 2);

        auto ot_fft = OTFFT::Factory::createComplexFFT(N);

        
        

        // bench.run("Kiss", [&]() {
        //     kiss_fft( cfg , (kiss_fft_cpx*) x ,  (kiss_fft_cpx*) y );
        // });

        bench.run("PGFFT", [&]() {
            pgfft.apply(x,  y); 
        });
        
        bench.run("OTFFT", [&]() {
            ot_fft->fwd((OTFFT::complex_t*)X); 
        });
 
        // bench.run("PFFFT", [&]() {
        //     pffftd_transform(ffts, (double*) x,  (double*) y, W, PFFFT_FORWARD);
        // });

        bench.run("AFFT", [&]() {
            fft.process_dit(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen, false, false, false);
        });

        bench.run("AFFT Slow", [&]() {
            fft_slow.process_dit(XX.data(), XX.data()+transformLen, ZZ.data(), ZZ.data()+transformLen);
        });

        bench.run("Ipp", [&]() {
            std::memcpy(pSrc, XX.data(), transformLen * 2 * sizeof(double));
            ippsFFTFwd_CToC_64fc(pSrc,pDst,pFFTSpec,pFFTWorkBuf);
        });
    }

    {
        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "New: " << transformLen;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        for (size_t i = 4; i <= 19; i++) {
            auto transformLen2 = 1 << i;
            auto XX = std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen2 * 2);
            auto ZZ =std::vector<double, xsimd::aligned_allocator<double, 128>>(transformLen2 * 2);
            FftComplex<StdSpec<double>, Double4Spec> fft2(transformLen2);
            ostringstream name;
            name << "FFT" << transformLen2;
            bench.run(name.str(), [&]() {
                fft2.process_dit(XX.data(), XX.data()+transformLen2, ZZ.data(), ZZ.data()+transformLen2, false, false, false);
            });
        }
    }
    
    //--------------------------------------------------------------------------

    std::size_t signal_len = 8;
    std::size_t spectra_len = (signal_len >> 1) + 1 ;
    FftReal<StdSpec<double>, Double4Spec> fft_real(signal_len);
    {
        std::vector<double> signal({1,22,3,4,1,22,3,4});
        std::vector<double> spectra_real(spectra_len);
        std::vector<double> spectra_imag(spectra_len);
        
        fft_real.compute_spectra(
            spectra_real.data(), 
            spectra_imag.data(), 
            signal.data()
        );

        for (int i = 0; i < spectra_len; i++)
        {
            std::cout << spectra_real[i] << ", " <<  spectra_imag[i] << std::endl;
        }
    }

    {
        std::vector<double> signal(signal_len);
        std::vector<double> spectra_real({61, 0, -5, 0, -43});
        std::vector<double> spectra_imag({0, 1, -36, -1, 0});

        fft_real.compute_signal(
            signal.data(),
            spectra_real.data(), 
            spectra_imag.data()
        );

        for (int i = 0; i < signal_len; i++)
        {
            std::cout << signal[i] << std::endl;
        }
    }

    // -------------------------------------------------------------------------

    {
        std::cout << "Convolution:" << std::endl;
        std::size_t signal_len_conv = 8;
        ConvolutionReal<StdSpec<double>, Double4Spec> conv(signal_len_conv);
        std::vector<double> signal({1,1,1,1,0,0,0,0});
        std::vector<double> signal_b({1,1,0,0,0,0,0,0});
        std::vector<double> signal_auto_conv(signal_len_conv);
        conv.compute_convolution(
            signal_auto_conv.data(),
            signal.data(),
            signal_b.data(),
            true
        );
        for (int i = 0; i < signal_len_conv; i++)
        {
            std::cout << signal_auto_conv[i] << std::endl;
        }
    }

    {
        ConvolutionReal<StdSpec<double>, Double4Spec> conv(transformLen);
        std::vector<double> signal(transformLen);
        std::vector<double> signal_b(transformLen);
        std::vector<double> signal_auto_conv(transformLen);
        std::vector<double> signal_auto_conv_b(transformLen);

        std::random_device rd;  // Non-deterministic random number generator
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::uniform_real_distribution<> dis(0.0, 1.0); 

        for (int i = 0; i < signal.size(); ++i) {
            signal[i] = dis(gen); 
            signal_b[i] = dis(gen);
        }

        ankerl::nanobench::Bench bench;
        ostringstream title_stream;
        title_stream << "Convolution: " << transformLen;
        bench.title(title_stream.str());

        bench.minEpochIterations(10);

        bench.run("Fast Convol", [&]() {
            conv.compute_convolution(
                signal_auto_conv.data(),
                signal.data(),
                signal_b.data(),
                true
            );
        });

        bench.run("Slow Convol", [&]() {
            conv.compute_convolution(
                signal_auto_conv.data(),
                signal.data(),
                signal_b.data(),
                false
            );
        });

        conv.compute_convolution(
            signal_auto_conv.data(),
            signal.data(),
            signal_b.data(),
            true
        );

        conv.compute_convolution(
            signal_auto_conv_b.data(),
            signal.data(),
            signal_b.data(),
            false
        );

        double conv_diff = 0;

        for (int i = 0; i < signal.size(); ++i) {
            conv_diff += abs(signal_auto_conv[i] - signal_auto_conv_b[i]);
        }

        cout << "Conv diff: " << conv_diff << endl;
    }

    pffftd_aligned_free(W);
    pffftd_aligned_free(Y);
    pffftd_aligned_free(X);
    pffftd_aligned_free(Z);
    pffftd_destroy_setup(ffts);
    kiss_fft_free(cfg);

    return 0;
}