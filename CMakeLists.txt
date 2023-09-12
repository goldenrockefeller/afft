cmake_minimum_required(VERSION 3.13.0)
project(afft VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(pffft REQUIRED) 
find_package(muFFT REQUIRED)  
find_package(xsimd REQUIRED)  

get_target_property(pffft_INCLUDE_DIRS PFFFT INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(mufft_INCLUDE_DIRS muFFT INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(xsimd_INCLUDE_DIRS xsimd INTERFACE_INCLUDE_DIRECTORIES)

add_executable(afft-run src/main.cpp src/PGFFT.cpp)
 
target_compile_options(afft-run PUBLIC /W2  /arch:AVX2 /fp:fast /EHsc /permissive-)
target_include_directories(afft-run PUBLIC ${pffft_INCLUDE_DIRS})
target_include_directories(afft-run PUBLIC ${mufft_INCLUDE_DIRS})
target_include_directories(afft-run PUBLIC ${xsimd_INCLUDE_DIRS})

target_link_libraries(afft-run PRIVATE PFFFT)
target_link_libraries(afft-run PRIVATE muFFT)

set_target_properties(afft-run PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO 
)  
 
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
