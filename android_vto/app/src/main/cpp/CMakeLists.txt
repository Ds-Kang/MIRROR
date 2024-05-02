
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("virtualtryon")

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

#
set(pathOPENCV ${CMAKE_CURRENT_SOURCE_DIR}/externlibs/libopencv)
set(pathLIBOPENCV_JAVA ${pathOPENCV}/libopencv_java4.so)
include_directories(${pathOPENCV})

set(pathONNX ${CMAKE_CURRENT_SOURCE_DIR}/externlibs/onnxruntime-release)
set(pathLIBONNX ${pathONNX}/jni/arm64-v8a/libonnxruntime.so)
include_directories(${pathONNX}/headers)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native_lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        nativeLib.cpp modelInference.cpp models.cpp modelUtils.cpp imageUtils.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${pathLIBOPENCV_JAVA})

add_library(lib_onnxruntime_c_api SHARED IMPORTED)
set_target_properties(lib_onnxruntime_c_api PROPERTIES IMPORTED_LOCATION ${pathLIBONNX})

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library( # Sets the name of the path variable.
        android-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        android)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries(  # Specifies the target library.
        native_lib
        lib_opencv
        lib_onnxruntime_c_api
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        ${android-lib})
