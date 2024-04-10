#include <iostream>
#include "fft_complex.hpp"
#include "fft_real.hpp"
#include "spec.hpp"
#include "pffft_double.h"
#include "fft.h"
#include "PGFFT.h"
#include "kiss_fft.h"
#include "nanobench.h"


using namespace std;
using namespace goldenrockefeller::afft;

template <typename Sample> 
struct OperandSpec{
    using Value = Sample[1];
};

int main() {
    constexpr std::size_t transformLen = 1 << 11;
    // 256, double, double, forward
    // 512, double, double, forward
    // 32, double, AVX, forward
    // 256, double AVX, forward
    // 512, double AVX, forward

    PFFFTD_Setup *ffts = pffftd_new_setup(transformLen, PFFFT_COMPLEX);
    PGFFT pgfft(transformLen);
    kiss_fft_cfg cfg=kiss_fft_alloc(transformLen,0,NULL,NULL);

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

    /* prepare some input data */
    for (int k = 0; k < transformLen; k += 2)
    {
        Z[k] =  k / 2;  /* real */
        Z[k+transformLen] = (k / 2) & 1;  /* imag */

        Z[k+1] = -1 - k / 2;  /* real */
        Z[k+1+transformLen] = (k / 2) & 1;  /* imag */
    }

    fft.ProcessDif(X, X+transformLen, Z, Z+transformLen);

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

    
    ankerl::nanobench::Bench bench;
    ostringstream title_stream;
    title_stream << "Size: " << transformLen;
    bench.title(title_stream.str());

    bench.minEpochIterations(10);

    bench.run("Kiss", [&]() {
        kiss_fft( cfg , (kiss_fft_cpx*) x ,  (kiss_fft_cpx*) y );
    });

    bench.run("PGFFT", [&]() {
        pgfft.apply(x,  y);
    });

    bench.run("PFFFT", [&]() {
        pffftd_transform(ffts, (double*) x,  (double*) y, W, PFFFT_FORWARD);
    });

    bench.run("AFFT", [&]() {
        fft.ProcessDit(X, X+transformLen, Z, Z+transformLen);
    });

    bench.run("AFFT Slow", [&]() {
        fft_slow.ProcessDit(X, X+transformLen, Z, Z+transformLen);
    });


    //--------------------------------------------------------------------------

    std::size_t signal_len = 8;
    std::size_t spectra_len = (signal_len >> 1) + 1 ;
    FftReal<StdSpec<double>, Double4Spec> fft_real(signal_len);
    {
        std::vector<double> signal({1,22,3,4,1,22,3,4});
        std::vector<double> spectra_real(spectra_len);
        std::vector<double> spectra_imag(spectra_len);
        
        fft_real.ComputeSpectra(
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

        fft_real.ComputeSignal(
            signal.data(),
            spectra_real.data(), 
            spectra_imag.data(),
            true
        );

        for (int i = 0; i < signal_len; i++)
        {
            std::cout << signal[i] << std::endl;
        }
    }

    pffftd_aligned_free(W);
    pffftd_aligned_free(Y);
    pffftd_aligned_free(X);
    pffftd_aligned_free(Z);
    pffftd_destroy_setup(ffts);
    kiss_fft_free(cfg);

    return 0;
}