cmake_minimum_required(VERSION 3.13.0)
project(afft VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(afft-run src/main.cpp)

target_compile_options(afft-run PUBLIC /W2  /arch:AVX2 /fp:fast /EHsc /permissive-)

set_target_properties(afft-run PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO 
) 
 
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
