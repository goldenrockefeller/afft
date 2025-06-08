cmake_minimum_required(VERSION 3.13.0)
project(afft VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(pffft REQUIRED) 
find_package(xsimd REQUIRED)  
find_package(IPP REQUIRED)
find_package(nanobench REQUIRED) 
find_package(otfft REQUIRED) 


add_library(RunTmChk STATIC src/RunTmChk.c)

add_executable(afft-run src/main.cpp src/PGFFT.cpp)

target_link_libraries(
   afft-run 
      PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kissfft-double.lib
   IPP::ippcore 
   IPP::ipps 
   IPP::ippi 
   IPP::ippcv
   PFFFT
   xsimd
   nanobench::nanobench
   RunTmChk
   otfft
)
 
# target_compile_options(afft-run PUBLIC /W2 /O2 /Oi /arch:AVX2 /fp:fast /EHsc /permissive-)
target_compile_options(afft-run PUBLIC
   -O3
   -ffast-math 
   -msse4.1
   -mavx 
   -mavx2 
   -mfma 
   -funroll-loops # --param max-unroll-times=8
   # -march=native
   )
target_include_directories( 
      afft-run 
   PUBLIC 
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> 
      $<INSTALL_INTERFACE:include> 
)

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
   TARGETS ${PROJECT_NAME}
   EXPORT ${PROJECT_NAME}-targets
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}-targets
   FILE ${PROJECT_NAME}Targets.cmake
   DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}
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

if(WIN32 AND MINGW AND NOT IPP_VERSION_MAJOR LESS 7)
    # Since Intel IPP built with Microsoft compiler and /GS option
    # ======================================================
    # From Windows SDK 7.1
    #   (usually in "C:\Program Files\Microsoft Visual Studio 10.0\VC\lib"),
    # to avoid undefined reference to __security_cookie and _chkstk:
    set(MSV_RUNTMCHK "RunTmChk")
    set(IPP_LIBRARIES ${IPP_LIBRARIES} ${MSV_RUNTMCHK}${IPP_LIB_SUFFIX})

    # To avoid undefined reference to _alldiv and _chkstk
    # ===================================================
    # NB: it may require a recompilation of w32api (after having modified
    #     the file ntdll.def) to export the required functions
    #     See http://code.opencv.org/issues/1906 for additional details
    set(MSV_NTDLL    "ntdll")
    set(IPP_LIBRARIES ${IPP_LIBRARIES} ${MSV_NTDLL}${IPP_LIB_SUFFIX})
endif()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
