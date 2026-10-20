// SPDX-License-Identifier: Apache-2.0
/**
 * @file    image_io.cpp
 * @brief   原尺寸 NV12 转换实现及紧密 stride 的只读视图封装。
 */
#include "image_io.hpp"

#include <opencv2/imgproc.hpp>

#include <cstring>
#include <limits>
#include <utility>

namespace rk_yolo::example {

/* 视图不拥有存储；由调用方保证 Nv12Image 在同步推理结束前一直有效。 */
FrameView Nv12Image::view() const noexcept
{
    FrameView frame;
    frame.bytes = ByteView(bytes.data(), bytes.size());
    frame.geometry = ImageGeometry{width, height, width, height};
    return frame;
}

bool bgrToNv12(const cv::Mat& bgr, Nv12Image& output, std::string& error)
{
    error.clear();
    if (bgr.empty() || bgr.dims != 2 || bgr.type() != CV_8UC3) {
        error = "Expected a nonempty 8-bit, three-channel BGR image.";
        return false;
    }
    /* NV12 4:2:0 的每个 UV 样本对应 2×2 像素，宽高均须为偶数。
     * 示例拒绝奇数输入，不悄悄裁剪/缩放，避免坐标与原图尺寸不一致。 */
    if ((bgr.cols & 1) != 0 || (bgr.rows & 1) != 0) {
        error = "NV12 requires even image width and height. This example does "
                "not resize the source; crop or pad the image to even dimensions first.";
        return false;
    }

    /* 先验证宽×高及加上半个 Y 平面的总容量不会溢出，再交给容器分配。 */
    const auto width = static_cast<std::size_t>(bgr.cols);
    const auto height = static_cast<std::size_t>(bgr.rows);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (width > maximum / height) {
        error = "Image dimensions overflow the NV12 buffer size.";
        return false;
    }
    const std::size_t yBytes = width * height;
    if (yBytes > maximum - yBytes / 2U) {
        error = "Image dimensions overflow the NV12 buffer size.";
        return false;
    }

    /* OpenCV 处理输入行跨度；随后将 I420 整理为连续存储，便于按平面拆分。
     * 转换产生 BT.601 limited 数值范围，不能用 full-range 矩阵解码。 */
    cv::Mat i420;
    cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    if (!i420.isContinuous())
        i420 = i420.clone();
    const std::size_t totalBytes = yBytes + yBytes / 2U;
    if (i420.type() != CV_8UC1 || i420.total() != totalBytes) {
        error = "Unexpected OpenCV I420 output layout.";
        return false;
    }

    /* 先保留 Y 平面原字节，再把平面 U、V 交织成 UVUV。
     * 输出 Y/UV 均无行尾填充，view() 据此使用原宽高作为 stride。 */
    Nv12Image converted;
    converted.width = bgr.cols;
    converted.height = bgr.rows;
    converted.bytes.resize(totalBytes);
    const auto* planar = i420.ptr<std::uint8_t>(0);
    std::memcpy(converted.bytes.data(), planar, yBytes);
    const auto* uPlane = planar + yBytes;
    const auto* vPlane = uPlane + yBytes / 4U;
    auto* uvPlane = converted.bytes.data() + yBytes;
    for (std::size_t i = 0; i < yBytes / 4U; ++i) {
        uvPlane[2U * i] = uPlane[i];
        uvPlane[2U * i + 1U] = vPlane[i];
    }
    /* 完整帧准备好后才提交所有权；旧 output 的缓冲在这里由容器正常释放。 */
    output = std::move(converted);
    return true;
}

} // namespace rk_yolo::example
