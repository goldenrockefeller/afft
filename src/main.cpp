#include <iostream>
#include "afft.hpp"
#include "spec.hpp"

using namespace std;
using namespace goldenrockefeller::afft;

template <typename Sample> 
struct OperandSpec{
    using Value = Sample[1];
};

int main() {
    Fft<8, StdSpec<double>, OperandSpec<double>> fft;

    cout << "std::size_t n_radix_4_butterflies;" << endl;
    cout << fft.n_radix_4_butterflies << endl;
    cout << "------------------------" << endl << endl;

    cout << "std::size_t n_radix_2_butterflies;" << endl;
    cout << fft.n_radix_2_butterflies << endl;
    cout << "------------------------" << endl << endl;

    cout << "bool using_final_radix_2_butterfly;" << endl;
    cout << fft.using_final_radix_2_butterfly << endl;
    cout << "------------------------" << endl << endl;

    cout << "std::vector<std::vector<std::vector<Sample>>> twiddles_real;" << endl;
    for (auto& twiddle : fft.twiddles_real){
        for (auto& subtwiddle : twiddle) {
            for (auto& elem : subtwiddle) {
                cout << elem << ",";
            }
            cout << endl; 
        }
        cout << endl;
    }
    cout << "------------------------" << endl << endl;

    cout << "std::vector<std::vector<std::vector<Sample>>> twiddles_imag;" << endl;
    for (auto& twiddle : fft.twiddles_imag){
        for (auto& subtwiddle : twiddle) {
            for (auto& elem : subtwiddle) {
                cout << elem << ",";
            }
            cout << endl; 
        }
        cout << endl;
    }
    cout << "------------------------" << endl << endl;

    cout << "std::vector<std::size_t> scrambled_indexes;" << endl;
    for (auto& id : fft.scrambled_indexes){
        cout << id << ",";
    }
    cout << endl;
    cout << "------------------------" << endl << endl;
    
    cout << "std::vector<std::size_t> scrambled_indexes_dft;" << endl;
    for (auto& id : fft.scrambled_indexes_dft){
        cout << id << ",";
    }
    cout << endl;
    cout << "------------------------" << endl << endl;
    
    cout << "std::vector<std::vector<Sample>> dft_real;" << endl;
    for (auto& dft_basis : fft.dft_real){
        for (auto& elem : dft_basis) {
            cout << elem << ",";
        }
        cout << endl;
    }
    cout << "------------------------" << endl << endl;

    cout << "std::vector<std::vector<Sample>> dft_imag;" << endl;
    for (auto& dft_basis : fft.dft_imag){
        for (auto& elem : dft_basis) {
            cout << elem << ",";
        }
        cout << endl;
    }
    cout << "------------------------" << endl << endl;

    return 0;
}