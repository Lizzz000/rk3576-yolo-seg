// SPDX-License-Identifier: Apache-2.0
/**
 * @file    image_io.hpp
 * @brief   示例输入转换：拥有 NV12 缓冲并向推理层提供只读帧视图。
 */
#ifndef RK_YOLO_EXAMPLES_IMAGE_IO_HPP
#define RK_YOLO_EXAMPLES_IMAGE_IO_HPP

#include "camera/camera_types.hpp"
#include "types/frame_types.hpp"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace rk_yolo::example {

/* 拥有连续紧密排列的 NV12 数据：先 Y 平面，再交错 UV，容量为宽×高×3/2 字节。
 * view() 只借用数据；推理完成前对象必须存活，且不得重分配、移动或改写 bytes。 */
struct Nv12Image final {
    std::vector<std::uint8_t> bytes;
    int width{0};
    int height{0};
    CameraColorSpace colorSpace{CameraColorSpace::Bt601Limited};

    /* 返回只读借用，不复制图像。Y 每行宽度为 width 字节，无额外 stride padding；
     * strideWidth=width、strideHeight=height，UV 起点为 width×height 字节。 */
    FrameView view() const noexcept;
};

/**
 * @brief  原尺寸 BGR→I420→NV12，不做缩放或 letterbox。
 * @param  bgr    非空 CV_8UC3 图像，宽高必须为偶数像素，可来自图片或 VideoCapture。
 * @param  output 成功时替换为转换结果；转换采用临时对象，失败时不提交半帧。
 * @param  error  普通参数/布局错误的文字说明，成功时为空。
 * @return true：完整 NV12 已准备好；false：输入、尺寸或输出布局不满足约束。
 * @note   OpenCV I420 为 BT.601 limited；U/V 交错仅改变存储排列，不改变数值范围，
 *         调用检测器时必须使用 output.colorSpace。OpenCV/内存异常交给入口处理。
 */
bool bgrToNv12(const cv::Mat& bgr, Nv12Image& output, std::string& error);

} // namespace rk_yolo::example

#endif
