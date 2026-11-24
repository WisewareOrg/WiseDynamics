# Builds the compiled Runge-Kutta integrator against bsk-sdk's headers of the exact Basilisk it links:
# the libdynamicsLib.so of the bsk wheel in the build environment.
cmake_minimum_required(VERSION 3.26)
project(wise_dynamics_native LANGUAGES NONE)

if(NOT (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"))
  message(FATAL_ERROR
    "wise-dynamics builds its integrator for Linux x86_64 only, not "
    "${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
enable_language(CXX)
# Only the flags below build the integrator: CXXFLAGS and the build type's flags would change its arithmetic.
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "")

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import bsk_sdk; print(bsk_sdk.cmake_config_dir())"
  OUTPUT_VARIABLE bsk-sdk_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
# Refuses a Basilisk of another version than the SDK's.
find_package(bsk-sdk CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import pathlib, Basilisk; print(pathlib.Path(Basilisk.__file__).parent); print(Basilisk.getBuildInfo()['artifact']['sourceRevision'])"
  OUTPUT_VARIABLE basilisk
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" basilisk "${basilisk}")
list(GET basilisk 0 basilisk_dir)
list(GET basilisk 1 basilisk_revision)

# Named as an extension module so the import system finds it in a wheel and an editable install alike;
# it defines no module and is loaded with ctypes.
add_library(runge_kutta SHARED native/runge_kutta.cpp)
set_target_properties(runge_kutta PROPERTIES PREFIX "" OUTPUT_NAME "_runge_kutta_native" SUFFIX ".abi3.so")
target_compile_features(runge_kutta PRIVATE cxx_std_17)
target_compile_definitions(runge_kutta PRIVATE NDEBUG WD_BASILISK_REVISION="${basilisk_revision}")
# The arithmetic and Eigen alignment of Basilisk's SSE2 build, whatever the compiler defaults to.
target_compile_options(runge_kutta PRIVATE
  -march=x86-64 -mtune=generic -O2 -fno-fast-math -ffp-contract=off -fvisibility=hidden -fvisibility-inlines-hidden)
target_include_directories(runge_kutta BEFORE PRIVATE native/include)
target_link_libraries(runge_kutta PRIVATE bsk::sdk_headers Eigen3::Eigen "${basilisk_dir}/libdynamicsLib.so")
install(TARGETS runge_kutta LIBRARY DESTINATION wise_dynamics/basilisk)
