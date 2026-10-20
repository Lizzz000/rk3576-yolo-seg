// SPDX-License-Identifier: Apache-2.0
/**
  * @file    input_validation.hpp
  * @brief   连续 NV12 帧的几何、存储长度与溢出检查
  * 用于进入 RGA 前验证内存边界；不检查图像内容或相机设备状态。
  */
#pragma once

#include "types/frame_types.hpp"

#include <limits>

namespace rk_yolo {

/* NV12 为 Y 平面后接交织 UV 平面，两者使用相同的行跨度。
 * 宽高及 stride 都须为正偶数且覆盖有效图像；成功时 required 返回总字节数，
 * 包含所有对齐填充。失败时 required=0，不进行可能溢出的尺寸乘法。 */
inline bool nv12RequiredBytes(ImageGeometry geometry, std::size_t& required) noexcept
{
    required = 0;
    if (!geometry.valid() || (geometry.width & 1) || (geometry.height & 1)
        || (geometry.strideWidth & 1) || (geometry.strideHeight & 1))
        return false;

    const auto strideWidth = static_cast<std::size_t>(geometry.strideWidth);
    const auto strideHeight = static_cast<std::size_t>(geometry.strideHeight);
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (strideWidth > maximum / strideHeight)
        return false;
    const auto lumaBytes = strideWidth * strideHeight; // Y 大小，UV 另占其一半
    if (lumaBytes > maximum - lumaBytes / 2)
        return false;
    required = lumaBytes + lumaBytes / 2;
    return true;
}

/* FrameView::valid() 仅检查非空与基本几何；此处进一步检查 NV12 偶数约束和
 * bytes.size()，避免把小缓冲区按完整图像交给 RGA 读取。 */
inline bool isValidNv12Frame(FrameView frame) noexcept
{
    std::size_t required = 0;
    return frame.valid() && nv12RequiredBytes(frame.geometry, required)
        && frame.bytes.size() >= required;
}

} // namespace rk_yolo
