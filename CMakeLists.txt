# @file  CMakeLists.txt
# @brief RK3576 推理库、图片示例和可选摄像头/显示示例的构建入口。
# 板端配置：-DRKNN_ROOT=<RKNN前缀> -DRGA_ROOT=<配套RGA前缀>。
# 模型与 SDK 由外部提供；生成文件统一放到独立的 build 目录。
cmake_minimum_required(VERSION 3.18)
project(rk3576_yolo_seg VERSION 0.1.0 LANGUAGES NONE)

# 在查找编译器前检查目标平台，避免在不支持的主机上尝试链接板端 SDK。
# 正式程序使用 RK3576 的真实 RKNN/RGA，目标系统须为 Linux ARM64。
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _target_processor)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
   OR NOT _target_processor MATCHES "^(aarch64|arm64)$")
  message(FATAL_ERROR
    "yolo_seg_image requires Linux/AArch64 on an RK3576 board. "
    "Detected target: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}.")
endif()
# 禁止源内构建，防止生成文件和复制的 SDK 头文件混入待发布源码。
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR "Use a separate build directory: cmake -S . -B build")
endif()
enable_language(CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 两个前缀可分别指定，但每个组件的头文件和库必须来自对应的同一前缀。
set(RKNN_ROOT "" CACHE PATH "RKNN SDK prefix containing include/rknn_api.h and the ARM64 runtime library")
set(RGA_ROOT "" CACHE PATH "Prefix containing a matched include/rga/im2d.h and librga")
foreach(_sdk_root IN ITEMS RKNN_ROOT RGA_ROOT)
  if("${${_sdk_root}}" STREQUAL "" OR NOT IS_DIRECTORY "${${_sdk_root}}")
    message(FATAL_ERROR
      "Set -D${_sdk_root}=<existing SDK prefix>. SDK files are not bundled.")
  endif()
endforeach()

# NO_DEFAULT_PATH 禁止静默混入另一套系统头文件/库；find_* 会复用缓存，
# 后续还会校验解析后的真实路径，切换 SDK 时应使用新的构建目录。
find_path(RKNN_INCLUDE_DIR rknn_api.h
  PATHS "${RKNN_ROOT}/include" NO_DEFAULT_PATH)
find_library(RKNN_LIBRARY NAMES rknnrt
  PATHS "${RKNN_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        "${RKNN_ROOT}/lib" "${RKNN_ROOT}/lib64" "${RKNN_ROOT}/aarch64"
  NO_DEFAULT_PATH)
find_path(RGA_INCLUDE_DIR rga/im2d.h
  PATHS "${RGA_ROOT}/include" NO_DEFAULT_PATH)
find_library(RGA_LIBRARY NAMES rga
  PATHS "${RGA_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        "${RGA_ROOT}/lib" "${RGA_ROOT}/lib64"
  NO_DEFAULT_PATH)

# 检查符号链接和缓存值解析后仍位于指定前缀内，避免只换头文件或只换库。
function(require_sdk_path variable root_variable)
  if(NOT DEFINED ${variable} OR NOT EXISTS "${${variable}}")
    message(FATAL_ERROR
      "Missing ${variable} under ${root_variable}=${${root_variable}}. "
      "Provide the matching ARM64 SDK headers and library, then configure again.")
  endif()
  get_filename_component(_root "${${root_variable}}" REALPATH)
  get_filename_component(_candidate "${${variable}}" REALPATH)
  string(REGEX REPLACE "/+$" "" _root "${_root}")
  string(FIND "${_candidate}/" "${_root}/" _prefix_position)
  if(NOT _prefix_position EQUAL 0)
    message(FATAL_ERROR
      "${variable}=${${variable}} resolves outside ${root_variable}=${${root_variable}}. "
      "Use a fresh build directory or clear the cached SDK paths.")
  endif()
endfunction()
require_sdk_path(RKNN_INCLUDE_DIR RKNN_ROOT)
require_sdk_path(RKNN_LIBRARY RKNN_ROOT)
require_sdk_path(RGA_INCLUDE_DIR RGA_ROOT)
require_sdk_path(RGA_LIBRARY RGA_ROOT)
if(IS_DIRECTORY "${RKNN_LIBRARY}" OR IS_DIRECTORY "${RGA_LIBRARY}")
  message(FATAL_ERROR "RKNN_LIBRARY and RGA_LIBRARY must identify library files, not directories.")
endif()
if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h"
   OR NOT EXISTS "${RGA_INCLUDE_DIR}/rga/im2d.h")
  message(FATAL_ERROR "Cached SDK include directories do not contain the required RKNN/RGA headers.")
endif()

# 某些 RKNN SDK 的 include 同时带有旧 RGA 头文件；这里只复制 rknn_api.h
# 到构建私有目录，不把整个 RKNN include 加入搜索路径，避免遮蔽所选 RGA。
set(_rknn_private_include "${CMAKE_CURRENT_BINARY_DIR}/rknn_headers")
file(MAKE_DIRECTORY "${_rknn_private_include}")
configure_file("${RKNN_INCLUDE_DIR}/rknn_api.h"
  "${_rknn_private_include}/rknn_api.h" COPYONLY)

# 默认构建图片和无窗口摄像头入口；窗口显示需显式启用并提供 highgui。
option(YOLO_BUILD_CAMERA "Build the OpenCV V4L2 camera example" ON)
option(YOLO_WITH_DISPLAY "Enable the optional camera window using OpenCV highgui" OFF)
if(YOLO_WITH_DISPLAY AND NOT YOLO_BUILD_CAMERA)
  message(FATAL_ERROR "YOLO_WITH_DISPLAY requires YOLO_BUILD_CAMERA=ON")
endif()
set(_opencv_components core imgproc imgcodecs)
if(YOLO_BUILD_CAMERA)
  list(APPEND _opencv_components videoio)
endif()
if(YOLO_WITH_DISPLAY)
  list(APPEND _opencv_components highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${_opencv_components})

# 核心只依赖 RKNN/RGA；图片、摄像头和标注工具在下方单独组织。
add_library(rk_yolo_core STATIC src/yolo/yolo.cpp)
target_compile_features(rk_yolo_core PUBLIC cxx_std_17)
set_target_properties(rk_yolo_core PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(rk_yolo_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(rk_yolo_core SYSTEM PRIVATE
  "${_rknn_private_include}" "${RGA_INCLUDE_DIR}")
target_link_libraries(rk_yolo_core PUBLIC "${RKNN_LIBRARY}" "${RGA_LIBRARY}" PRIVATE m)

# 两个入口复用相同的标签校验、NV12 转换及结果显示，避免维护两套规则。
add_library(rk_yolo_example_support STATIC examples/image_io.cpp examples/shared_options.cpp)
target_compile_features(rk_yolo_example_support PUBLIC cxx_std_17)
set_target_properties(rk_yolo_example_support PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(rk_yolo_example_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/examples")
target_include_directories(rk_yolo_example_support SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rk_yolo_example_support PUBLIC rk_yolo_core ${OpenCV_LIBS})

add_executable(yolo_seg_image examples/image_demo.cpp)
target_link_libraries(yolo_seg_image PRIVATE rk_yolo_example_support)
set_target_properties(yolo_seg_image PROPERTIES CXX_EXTENSIONS OFF)
if(YOLO_BUILD_CAMERA)
  add_executable(yolo_seg_camera examples/camera_demo.cpp)
  target_link_libraries(yolo_seg_camera PRIVATE rk_yolo_example_support)
  set_target_properties(yolo_seg_camera PROPERTIES CXX_EXTENSIONS OFF)
  if(YOLO_WITH_DISPLAY)
    target_compile_definitions(yolo_seg_camera PRIVATE RK_YOLO_WITH_HIGHGUI=1)
  endif()
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rk_yolo_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rk_yolo_example_support PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(yolo_seg_image PRIVATE -Wall -Wextra -Wpedantic)
  if(YOLO_BUILD_CAMERA)
    target_compile_options(yolo_seg_camera PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# 配置时打印最终命中的依赖，便于确认实际使用的 SDK 版本和安装位置。
message(STATUS "RKNN header: ${RKNN_INCLUDE_DIR}/rknn_api.h")
message(STATUS "RKNN runtime: ${RKNN_LIBRARY}")
message(STATUS "RGA headers: ${RGA_INCLUDE_DIR}; library: ${RGA_LIBRARY}")
