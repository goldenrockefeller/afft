#ifndef GOLDENROCEKEFELLER_AFFT_SPEC
#define GOLDENROCEKEFELLER_AFFT_SPEC

#include <cmath>

#include "xsimd/xsimd.hpp"

namespace afft{

    template <typename ValueT>
    struct StdSpec{
        using Value = ValueT;

        static inline Value cos(const Value& x) {
            return std::cos(x);
        }

        static inline Value sin(const Value& x) {
            return std::sin(x);
        }

        static inline Value fma(const Value& x, const Value& y, const Value& z) {
            return x*y + z;
        }

        static inline Value fms(const Value& x, const Value& y, const Value& z) {
            return x*y - z;
        }

        static inline Value pi() {
            return Value(3.14159265358979323846264338327950288L);
        }

        template <typename SampleValue>
        static inline void load(const SampleValue* t, Value& x) {
            x = *t;
        }

        static inline void store(Value* t, const Value& x) {
            *t = x;
        }
    };

    struct Double4Spec{
        using Value = xsimd::batch<double, xsimd::avx>;

        static inline void load(const double* ptr, Value& x) {
            x = xsimd::batch<double, xsimd::avx>::load_unaligned(ptr);
        }

        static inline void store(double* ptr, const Value& x) {
            x.store_unaligned(ptr);
        }
    
        static inline Value fma(const Value& x, const Value& y, const Value& z) {
            return xsimd::fma(x, y, z);
        }

        static inline Value fms(const Value& x, const Value& y, const Value& z) {
            return xsimd::fms(x, y, z);
        }

        static inline void transpose_diagonal(double* real, double* imag, const std::size_t* indexes) {
            Value real_a;
            Value real_b;
            Value real_c;
            Value real_d;

            Value imag_a;
            Value imag_b;
            Value imag_c;
            Value imag_d;

            Value real_tmp_a;
            Value real_tmp_b;
            Value real_tmp_c;
            Value real_tmp_d;

            Value imag_tmp_a;
            Value imag_tmp_b;
            Value imag_tmp_c;
            Value imag_tmp_d;

            size_t i0 = indexes[0];
            size_t i1 = indexes[1];
            size_t i2 = indexes[2];
            size_t i3 = indexes[3];
        

            real_a = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i0);
            real_b = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i1);
            real_c = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i2);
            real_d = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i3);

            imag_a = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i0);
            imag_b = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i1);
            imag_c = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i2);
            imag_d = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i3);

            real_tmp_a = xsimd::zip_lo(real_a, real_c);
            real_tmp_b = xsimd::zip_hi(real_a, real_c);
            real_tmp_c = xsimd::zip_lo(real_b, real_d);
            real_tmp_d = xsimd::zip_hi(real_b, real_d);

            imag_tmp_a = xsimd::zip_lo(imag_a, imag_c);
            imag_tmp_b = xsimd::zip_hi(imag_a, imag_c);
            imag_tmp_c = xsimd::zip_lo(imag_b, imag_d);
            imag_tmp_d = xsimd::zip_hi(imag_b, imag_d);

            real_a = xsimd::zip_lo(real_tmp_a, real_tmp_c);
            real_b = xsimd::zip_hi(real_tmp_a, real_tmp_c);
            real_c = xsimd::zip_lo(real_tmp_b, real_tmp_d);
            real_d = xsimd::zip_hi(real_tmp_b, real_tmp_d);

            imag_a = xsimd::zip_lo(imag_tmp_a, imag_tmp_c);
            imag_b = xsimd::zip_hi(imag_tmp_a, imag_tmp_c);
            imag_c = xsimd::zip_lo(imag_tmp_b, imag_tmp_d);
            imag_d = xsimd::zip_hi(imag_tmp_b, imag_tmp_d);

            real_a.store_unaligned(real + i0);
            real_b.store_unaligned(real + i1);
            real_c.store_unaligned(real + i2);
            real_d.store_unaligned(real + i3);

            imag_a.store_unaligned(imag + i0);
            imag_b.store_unaligned(imag + i1);
            imag_c.store_unaligned(imag + i2);
            imag_d.store_unaligned(imag + i3);
        }

        static inline void transpose_off_diagonal(double* real, double* imag, const std::size_t* indexes) {
            Value row_a_a;
            Value row_a_b;
            Value row_a_c;
            Value row_a_d;

            Value row_b_a;
            Value row_b_b;
            Value row_b_c;
            Value row_b_d;

            Value row_a_tmp_a;
            Value row_a_tmp_b;
            Value row_a_tmp_c;
            Value row_a_tmp_d;

            Value row_b_tmp_a;
            Value row_b_tmp_b;
            Value row_b_tmp_c;
            Value row_b_tmp_d;

            size_t i0 = indexes[0];
            size_t i1 = indexes[1];
            size_t i2 = indexes[2];
            size_t i3 = indexes[3];
            size_t i4 = indexes[4];
            size_t i5 = indexes[5];
            size_t i6 = indexes[6];
            size_t i7 = indexes[7];

            row_a_a = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i0);
            row_a_b = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i1);
            row_a_c = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i2);
            row_a_d = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i3);

            row_b_a = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i4);
            row_b_b = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i5);
            row_b_c = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i6);
            row_b_d = xsimd::batch<double, xsimd::avx>::load_unaligned(real + i7);

            row_a_tmp_a = xsimd::zip_lo(row_a_a, row_a_c);
            row_a_tmp_b = xsimd::zip_hi(row_a_a, row_a_c);
            row_a_tmp_c = xsimd::zip_lo(row_a_b, row_a_d);
            row_a_tmp_d = xsimd::zip_hi(row_a_b, row_a_d);

            row_b_tmp_a = xsimd::zip_lo(row_b_a, row_b_c);
            row_b_tmp_b = xsimd::zip_hi(row_b_a, row_b_c);
            row_b_tmp_c = xsimd::zip_lo(row_b_b, row_b_d);
            row_b_tmp_d = xsimd::zip_hi(row_b_b, row_b_d);

            row_a_a = xsimd::zip_lo(row_a_tmp_a, row_a_tmp_c);
            row_a_b = xsimd::zip_hi(row_a_tmp_a, row_a_tmp_c);
            row_a_c = xsimd::zip_lo(row_a_tmp_b, row_a_tmp_d);
            row_a_d = xsimd::zip_hi(row_a_tmp_b, row_a_tmp_d);

            row_b_a = xsimd::zip_lo(row_b_tmp_a, row_b_tmp_c);
            row_b_b = xsimd::zip_hi(row_b_tmp_a, row_b_tmp_c);
            row_b_c = xsimd::zip_lo(row_b_tmp_b, row_b_tmp_d);
            row_b_d = xsimd::zip_hi(row_b_tmp_b, row_b_tmp_d);

            row_a_a.store_unaligned(real + i4);
            row_a_b.store_unaligned(real + i5);
            row_a_c.store_unaligned(real + i6);
            row_a_d.store_unaligned(real + i7);

            row_b_a.store_unaligned(real + i0);
            row_b_b.store_unaligned(real + i1);
            row_b_c.store_unaligned(real + i2);
            row_b_d.store_unaligned(real + i3);

            row_a_a = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i0);
            row_a_b = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i1);
            row_a_c = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i2);
            row_a_d = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i3);

            row_b_a = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i4);
            row_b_b = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i5);
            row_b_c = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i6);
            row_b_d = xsimd::batch<double, xsimd::avx>::load_unaligned(imag + i7);

            row_a_tmp_a = xsimd::zip_lo(row_a_a, row_a_c);
            row_a_tmp_b = xsimd::zip_hi(row_a_a, row_a_c);
            row_a_tmp_c = xsimd::zip_lo(row_a_b, row_a_d);
            row_a_tmp_d = xsimd::zip_hi(row_a_b, row_a_d);

            row_b_tmp_a = xsimd::zip_lo(row_b_a, row_b_c);
            row_b_tmp_b = xsimd::zip_hi(row_b_a, row_b_c);
            row_b_tmp_c = xsimd::zip_lo(row_b_b, row_b_d);
            row_b_tmp_d = xsimd::zip_hi(row_b_b, row_b_d);

            row_a_a = xsimd::zip_lo(row_a_tmp_a, row_a_tmp_c);
            row_a_b = xsimd::zip_hi(row_a_tmp_a, row_a_tmp_c);
            row_a_c = xsimd::zip_lo(row_a_tmp_b, row_a_tmp_d);
            row_a_d = xsimd::zip_hi(row_a_tmp_b, row_a_tmp_d);

            row_b_a = xsimd::zip_lo(row_b_tmp_a, row_b_tmp_c);
            row_b_b = xsimd::zip_hi(row_b_tmp_a, row_b_tmp_c);
            row_b_c = xsimd::zip_lo(row_b_tmp_b, row_b_tmp_d);
            row_b_d = xsimd::zip_hi(row_b_tmp_b, row_b_tmp_d);

            row_a_a.store_unaligned(imag + i4);
            row_a_b.store_unaligned(imag + i5);
            row_a_c.store_unaligned(imag + i6);
            row_a_d.store_unaligned(imag + i7);

            row_b_a.store_unaligned(imag + i0);
            row_b_b.store_unaligned(imag + i1);
            row_b_c.store_unaligned(imag + i2);
            row_b_d.store_unaligned(imag + i3);
        }

    };

}



#endif