#ifndef GOLDENROCEKEFELLER_AFFT_SPEC
#define GOLDENROCEKEFELLER_AFFT_SPEC

#include <cmath>

#include "xsimd/xsimd.hpp"



// template<typename Sample, typename OperandSpec>
// static inline void proto_transpose_off_diagonal(Sample* real, Sample* imag, const std::size_t* indexes) {
//     using Operand = OperandSpec::Value;
//     constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
//     Value row_x[n_samples_per_operand];
//     Value row_y[n_samples_per_operand];
//     Value row_u[n_samples_per_operand];
//     Value row_v[n_samples_per_operand];

//     // Value row_x[0];
//     // Value row_x[1];
//     // Value row_x[2];
//     // Value row_x[3];

//     // Value row_y[0];
//     // Value row_y[1];
//     // Value row_y[2];
//     // Value row_y[3];

//     // Value row_u[0];
//     // Value row_u[1];
//     // Value row_u[2];
//     // Value row_u[3];

//     // Value row_v[0];
//     // Value row_v[1];
//     // Value row_v[2];
//     // Value row_v[3];

//     std::size_t ind[2 * n_samples_per_operand];

//     copy_size_t_array<0, 8>(ind, indexes);
//     size_t i1 = indexes[1];
//     size_t i2 = indexes[2];
//     size_t i3 = indexes[3];
//     size_t i4 = indexes[4];
//     size_t i5 = indexes[5];
//     size_t i6 = indexes[6];
//     size_t i7 = indexes[7];

//     row_x[0] = xsimd::batch<double, arch>::load_unaligned(real + i0);
//     row_x[1] = xsimd::batch<double, arch>::load_unaligned(real + i1);
//     row_x[2] = xsimd::batch<double, arch>::load_unaligned(real + i2);
//     row_x[3] = xsimd::batch<double, arch>::load_unaligned(real + i3);

//     row_y[0] = xsimd::batch<double, arch>::load_unaligned(real + i4);
//     row_y[1] = xsimd::batch<double, arch>::load_unaligned(real + i5);
//     row_y[2] = xsimd::batch<double, arch>::load_unaligned(real + i6);
//     row_y[3] = xsimd::batch<double, arch>::load_unaligned(real + i7);

//     row_u[0] = xsimd::zip_lo(row_x[0], row_x[2]);
//     row_u[1] = xsimd::zip_hi(row_x[0], row_x[2]);
//     row_u[2] = xsimd::zip_lo(row_x[1], row_x[3]);
//     row_u[3] = xsimd::zip_hi(row_x[1], row_x[3]);

//     row_v[0] = xsimd::zip_lo(row_y[0], row_y[2]);
//     row_v[1] = xsimd::zip_hi(row_y[0], row_y[2]);
//     row_v[2] = xsimd::zip_lo(row_y[1], row_y[3]);
//     row_v[3] = xsimd::zip_hi(row_y[1], row_y[3]);

//     row_x[0] = xsimd::zip_lo(row_u[0], row_u[2]);
//     row_x[1] = xsimd::zip_hi(row_u[0], row_u[2]);
//     row_x[2] = xsimd::zip_lo(row_u[1], row_u[3]);
//     row_x[3] = xsimd::zip_hi(row_u[1], row_u[3]);

//     row_y[0] = xsimd::zip_lo(row_v[0], row_v[2]);
//     row_y[1] = xsimd::zip_hi(row_v[0], row_v[2]);
//     row_y[2] = xsimd::zip_lo(row_v[1], row_v[3]);
//     row_y[3] = xsimd::zip_hi(row_v[1], row_v[3]);

//     row_x[0].store_unaligned(real + i4);
//     row_x[1].store_unaligned(real + i5);
//     row_x[2].store_unaligned(real + i6);
//     row_x[3].store_unaligned(real + i7);

//     row_y[0].store_unaligned(real + i0);
//     row_y[1].store_unaligned(real + i1);
//     row_y[2].store_unaligned(real + i2);
//     row_y[3].store_unaligned(real + i3);

//     row_x[0] = xsimd::batch<double, arch>::load_unaligned(imag + i0);
//     row_x[1] = xsimd::batch<double, arch>::load_unaligned(imag + i1);
//     row_x[2] = xsimd::batch<double, arch>::load_unaligned(imag + i2);
//     row_x[3] = xsimd::batch<double, arch>::load_unaligned(imag + i3);

//     row_y[0] = xsimd::batch<double, arch>::load_unaligned(imag + i4);
//     row_y[1] = xsimd::batch<double, arch>::load_unaligned(imag + i5);
//     row_y[2] = xsimd::batch<double, arch>::load_unaligned(imag + i6);
//     row_y[3] = xsimd::batch<double, arch>::load_unaligned(imag + i7);

//     row_u[0] = xsimd::zip_lo(row_x[0], row_x[2]);
//     row_u[1] = xsimd::zip_hi(row_x[0], row_x[2]);
//     row_u[2] = xsimd::zip_lo(row_x[1], row_x[3]);
//     row_u[3] = xsimd::zip_hi(row_x[1], row_x[3]);

//     row_v[0] = xsimd::zip_lo(row_y[0], row_y[2]);
//     row_v[1] = xsimd::zip_hi(row_y[0], row_y[2]);
//     row_v[2] = xsimd::zip_lo(row_y[1], row_y[3]);
//     row_v[3] = xsimd::zip_hi(row_y[1], row_y[3]);

//     row_x[0] = xsimd::zip_lo(row_u[0], row_u[2]);
//     row_x[1] = xsimd::zip_hi(row_u[0], row_u[2]);
//     row_x[2] = xsimd::zip_lo(row_u[1], row_u[3]);
//     row_x[3] = xsimd::zip_hi(row_u[1], row_u[3]);

//     row_y[0] = xsimd::zip_lo(row_v[0], row_v[2]);
//     row_y[1] = xsimd::zip_hi(row_v[0], row_v[2]);
//     row_y[2] = xsimd::zip_lo(row_v[1], row_v[3]);
//     row_y[3] = xsimd::zip_hi(row_v[1], row_v[3]);

//     row_x[0].store_unaligned(imag + i4);
//     row_x[1].store_unaligned(imag + i5);
//     row_x[2].store_unaligned(imag + i6);
//     row_x[3].store_unaligned(imag + i7);

//     row_y[0].store_unaligned(imag + i0);
//     row_y[1].store_unaligned(imag + i1);
//     row_y[2].store_unaligned(imag + i2);
//     row_y[3].store_unaligned(imag + i3);
// }

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
        using Value = xsimd::batch<double, xsimd::fma3<xsimd::avx2>>;
        using arch = typename xsimd::fma3<xsimd::avx2>;

        static inline void load(const double* ptr, Value& x) {
            x = xsimd::batch<double, arch>::load_unaligned(ptr);
        }

        static inline void store(double* ptr, const Value& x) {
            x.store_unaligned(ptr);
        }
    
        static inline Value fma(const Value& x, const Value& y, const Value& z) {
            return xsimd::fma(x, y, z);
        }

        static inline Value fnma(const Value& x, const Value& y, const Value& z) {
            return xsimd::fnma(x, y, z);
        }

        static inline Value fms(const Value& x, const Value& y, const Value& z) {
            return xsimd::fms(x, y, z);
        }

        static inline void interleave(Value& x, Value& y, const Value& u, const Value& v) {
            x = xsimd::zip_lo(u, v);
            y = xsimd::zip_hi(u, v);
        }
    };

}



#endif