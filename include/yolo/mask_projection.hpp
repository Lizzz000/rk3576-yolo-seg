// SPDX-License-Identifier: Apache-2.0
/**
  * @file    mask_projection.hpp
  * @brief   INT8 prototype 的逐行整数投影
  * 输出为 sigmoid 之前的整数点积。量化 scale 为正时，可用其符号判断 mask>0.5。
  */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// 保留32通道完整整数点积，只调整遍历顺序；不降采样、不跳目标、不改变mask阈值。
class MaskProjection final {
public:
  /* proto 为 NCHW，plane 为单通道平面的元素数，offset 为该平面内的起始下标。
   * coeff 为已扣除系数零点的权重；zeroPoint 为 prototype 的量化零点。
   * 计算 channels 个通道、连续 count 个像素，写入至少 count 个 int32 元素。
   * 调用者保证 count 不跨行、索引有效、输出不与输入重叠；本函数不分配内存。
   * 当前使用 32 通道 INT8 数据及其零点，累加范围由上层模型契约保证。 */
  static void int8Row(const std::int8_t* proto, const std::int16_t* coeff,
                     std::size_t channels, std::size_t plane,
                     std::size_t offset, std::size_t count, int zeroPoint,
                     std::int32_t* output) noexcept
  {
    // 先合并每个通道的 prototype 零点补偿，避免在像素内层反复相减。
    std::int32_t bias = 0;
    for (std::size_t k = 0; k < channels; ++k)
      bias -= static_cast<std::int32_t>(coeff[k]) * zeroPoint;
    std::fill_n(output, count, bias);
    for (std::size_t k = 0; k < channels; ++k) {
      const auto* row = proto + k * plane + offset;
      const std::int32_t weight = coeff[k];
      // 连续访问同一通道的一行，避免逐像素跨32个大平面跳读。
      // 每个x对应独立累加器，Release编译器可自动向量化。
      for (std::size_t x = 0; x < count; ++x)
        output[x] += weight * static_cast<std::int32_t>(row[x]);
    }
  }
};
