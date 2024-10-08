cmake_minimum_required(VERSION 3.13.0)
project(afft VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(pffft REQUIRED) 
find_package(muFFT REQUIRED)  
find_package(xsimd REQUIRED)  
find_package(IPP REQUIRED)

get_target_property(pffft_INCLUDE_DIRS PFFFT INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(mufft_INCLUDE_DIRS muFFT INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(xsimd_INCLUDE_DIRS xsimd INTERFACE_INCLUDE_DIRECTORIES)

find_package(nanobench REQUIRED) 
get_target_property(nanobench_INCLUDE_DIRS nanobench::nanobench INTERFACE_INCLUDE_DIRECTORIES)

# add_library(pocketfft src/pocketfft.c)
add_executable(afft-run src/main.cpp src/PGFFT.cpp)

target_link_libraries(
   afft-run 
      PRIVATE 
   C:/Users/white/OneDrive/Documents/Workspace/afft/src/kissfft-double.lib
   IPP::ippcore 
   IPP::ipps 
   IPP::ippi 
   IPP::ippcv
)
 
target_compile_options(afft-run PUBLIC /W2 /O2 /arch:AVX2 /fp:fast /EHsc /permissive-)
target_include_directories(
      afft-run 
   PUBLIC 
      ${CMAKE_CURRENT_SOURCE_DIR}//include
      ${pffft_INCLUDE_DIRS} 
      ${nanobench_INCLUDE_DIRS}
      ${mufft_INCLUDE_DIRS}
      ${xsimd_INCLUDE_DIRS})

target_link_libraries(afft-run PRIVATE PFFFT muFFT nanobench::nanobench)

set_target_properties(afft-run PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO 
)  

# CREATE HEADER ONLY LIBRARY
# ---------------------------------------------------
add_library(afft INTERFACE)
target_include_directories(afft INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)  


# INSTALLATION 
# ---------------------------------------------------

include(GNUInstallDirs)

install(
   TARGETS afft
   EXPORT ${PROJECT_NAME}-targets
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}-targets
   FILE ${PROJECT_NAME}Targets.cmake
   NAMESPACE goldenrockefeller::
   DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/afft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY SameMajorVersion)

configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake
    )
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake
    )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
