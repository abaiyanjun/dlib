# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_definitions( -DDLIB_NO_GUI_SUPPORT -DLIB_PNG_SUPPORT -DLIB_JPEG_SUPPORT)

#set(ANDROID_OPENCV_COMPONENTS "opencv_java" CACHE STRING "")
#message(STATUS "ANDROID_ABI=${ANDROID_ABI}")
#find_package(OpenCV REQUIRED COMPONENTS ${ANDROID_OPENCV_COMPONENTS})


#jni头文件
include_directories(. dlib1919 jni_common jni_detections opencv42 ${CMAKE_CURRENT_LIST_DIR})

#导入库文件
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

file(GLOB source_files
        logging.cpp
        jni_common/jni_bitmap2mat.cpp
        jni_common/jni_fileutils.cpp
        jni_common/jni_utils.cpp
        jni_common/rgb2yuv.cpp
        jni_common/yuv2rgb.cpp
        jni_detections/jni_face_det.cpp
        jni_detections/jni_imageutils.cpp
        jni_detections/jni_pedestrian_det.cpp
        DetectionBasedTracker_jni.cpp
        )


add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             native-lib.cpp ${source_files} )


add_library( # Sets the name of the library.
        dlib-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        dlib1919/dlib/all/source.cpp
        #dlib1919/dlib/threads/threads_kernel_shared.cpp
        #dlib1919/dlib/entropy_encoder/entropy_encoder_kernel_1.cpp
        #dlib1919/dlib/entropy_decoder/entropy_decoder_kernel_2.cpp
        #dlib1919/dlib/base64/base64_kernel_1.cpp
        #dlib1919/dlib/threads/threads_kernel_1.cpp
        #dlib1919/dlib/threads/threads_kernel_2.cpp
        )


# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib
                       dlib-lib
                        -ljnigraphics
                        #添加opencv_java4
                        opencv_java4
                       # ${ANDROID_OPENCV_COMPONENTS}
                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )