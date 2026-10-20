// SPDX-License-Identifier: Apache-2.0
/**
  * @file    camera_types.hpp
  * @brief   相机来源、像素格式与 NV12 色彩范围的值类型
  * 枚举本身不执行设备发现或格式协商；当前摄像头示例由 OpenCV/V4L2 取流。
  */
#ifndef RK3576_YOLO_SEG_CAMERA_TYPES_H
#define RK3576_YOLO_SEG_CAMERA_TYPES_H

namespace rk_yolo
{

/* 供外部相机适配器描述偏好，不表示本示例自动配置 CSI 媒体链路。 */
enum class CameraPreference : int
{
    Auto = 0,
    Csi = 1,
    Usb = 2
};

/* 相机采集后端标识，与输入图像实际采用的像素格式分开记录。 */
enum class CameraBackend : int
{
    Unknown = 0,
    CsiRkisp,
    UsbUvc,
    OtherV4l2
};

/* 原始采集格式；进入 YoloDetector::detect() 前须准备连续 NV12。 */
enum class CameraSourceFormat : int
{
    Unknown = 0,
    Nv12,
    Yuyv,
    Mjpeg
};

/* NV12 的布局不决定其色彩范围。需按真实帧设置，不能把 limited 当 full。
 * OpenCV BGR→I420→NV12 路径使用 Bt601Limited；RGA 据此选择转换矩阵。 */
enum class CameraColorSpace : int
{
    Bt601Full = 0,
    Bt601Limited,
    Bt709Limited
};

} // namespace rk_yolo

/* 供底层适配代码使用的同值别名，不引入额外的状态或转换。 */
using CameraPreference_t = rk_yolo::CameraPreference;
using CameraBackend_t = rk_yolo::CameraBackend;
using CameraSourceFormat_t = rk_yolo::CameraSourceFormat;
using CameraColorSpace_t = rk_yolo::CameraColorSpace;

inline constexpr CameraPreference_t CAMERA_PREFERENCE_AUTO = CameraPreference_t::Auto;
inline constexpr CameraPreference_t CAMERA_PREFERENCE_CSI = CameraPreference_t::Csi;
inline constexpr CameraPreference_t CAMERA_PREFERENCE_USB = CameraPreference_t::Usb;

inline constexpr CameraBackend_t CAMERA_BACKEND_UNKNOWN = CameraBackend_t::Unknown;
inline constexpr CameraBackend_t CAMERA_BACKEND_CSI_RKISP = CameraBackend_t::CsiRkisp;
inline constexpr CameraBackend_t CAMERA_BACKEND_USB_UVC = CameraBackend_t::UsbUvc;
inline constexpr CameraBackend_t CAMERA_BACKEND_OTHER_V4L2 = CameraBackend_t::OtherV4l2;

inline constexpr CameraSourceFormat_t CAMERA_SOURCE_UNKNOWN = CameraSourceFormat_t::Unknown;
inline constexpr CameraSourceFormat_t CAMERA_SOURCE_NV12 = CameraSourceFormat_t::Nv12;
inline constexpr CameraSourceFormat_t CAMERA_SOURCE_YUYV = CameraSourceFormat_t::Yuyv;
inline constexpr CameraSourceFormat_t CAMERA_SOURCE_MJPEG = CameraSourceFormat_t::Mjpeg;

inline constexpr CameraColorSpace_t CAMERA_COLOR_BT601_FULL = CameraColorSpace_t::Bt601Full;
inline constexpr CameraColorSpace_t CAMERA_COLOR_BT601_LIMITED = CameraColorSpace_t::Bt601Limited;
inline constexpr CameraColorSpace_t CAMERA_COLOR_BT709_LIMITED = CameraColorSpace_t::Bt709Limited;

#endif
