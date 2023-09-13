#include <iostream>
#include "afft.hpp"
#include "spec.hpp"
#include "pffft_double.h"
#include "fft.h"
#include "PGFFT.h"
#include "kiss_fft.h"


using namespace std;
using namespace goldenrockefeller::afft;

template <typename Sample> 
struct OperandSpec{
    using Value = Sample[1];
};

int main() {
    constexpr std::size_t transformLen = 32;
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

    FftComplex<transformLen, StdSpec<double>, Double4Spec> fft;

    /* prepare some input data */
    for (int k = 0; k < transformLen; k += 2)
    {
        Z[k] =  k / 2;  /* real */
        Z[k+transformLen] = (k / 2) & 1;  /* imag */

        Z[k+1] = -1 - k / 2;  /* real */
        Z[k+1+transformLen] = (k / 2) & 1;  /* imag */
    }

    fft.Process<false>(X, X+transformLen, Z, Z+transformLen);

    /* compare output data */
    double diff = 0;
    for (int k = 0; k < transformLen; k += 1)
    {
        double new_diff = abs(Y[2 * k] - X[k]) + abs(Y[2 * k + 1] - X[k + transformLen]);

        cout << X[k] << " " << X[k + transformLen] << endl;

        if (new_diff > diff) {
            diff = new_diff;
        }
    }

    cout << diff << endl;

    pffftd_aligned_free(W);
    pffftd_aligned_free(Y);
    pffftd_aligned_free(X);
    pffftd_aligned_free(Z);
    pffftd_destroy_setup(ffts);
    kiss_fft_free(cfg);

    return 0;
}