// SPDX-License-Identifier: Apache-2.0
/**
  * @file    frame_types.hpp
  * @brief   图像存储视图、有效尺寸与帧标识
  * 这里只描述外部内存，不分配或释放图像。输入推理前还须验证 NV12 存储大小。
  */
#ifndef RK3576_YOLO_SEG_FRAME_TYPES_H
#define RK3576_YOLO_SEG_FRAME_TYPES_H

#include <cstddef>
#include <cstdint>

namespace rk_yolo
{

/* 只读字节视图：size 的单位为字节，内存由生产者持有。
 * 拷贝视图不会复制像素；operator[] 不做边界检查，调用方须保证下标有效。 */
class ByteView final
{
public:
    constexpr ByteView() noexcept = default;
    constexpr ByteView(const std::uint8_t* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    constexpr const std::uint8_t* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return data_ == nullptr || size_ == 0; }
    constexpr const std::uint8_t& operator[](std::size_t index) const noexcept
    {
        return data_[index];
    }

private:
    const std::uint8_t* data_{nullptr};
    std::size_t size_{0};
};

/* 可写字节视图，供拥有缓冲区的对象或硬件适配层使用；同样不转移所有权。 */
class MutableByteView final
{
public:
    constexpr MutableByteView() noexcept = default;
    constexpr MutableByteView(std::uint8_t* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    constexpr std::uint8_t* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return data_ == nullptr || size_ == 0; }
    constexpr std::uint8_t& operator[](std::size_t index) const noexcept
    {
        return data_[index];
    }

private:
    std::uint8_t* data_{nullptr};
    std::size_t size_{0};
};

struct ImageGeometry final
{
    // 有效图像宽高，单位为像素；不能把对齐填充算入检测坐标。
    int width{0};
    int height{0};
    // 分配的行跨度/行数，可包含对齐填充。对 NV12，Y 平面每像素占一字节。
    int strideWidth{0};
    int strideHeight{0};

    constexpr bool valid() const noexcept
    {
        return width > 0 && height > 0 && strideWidth >= width
            && strideHeight >= height;
    }
};

struct FrameView final
{
    // 本项目推理使用连续 NV12：Y 平面在前、交织 UV 在后。
    ByteView bytes{};
    ImageGeometry geometry{};
    // 可选帧标识与毫秒时间戳；-1 表示时间未提供，时钟基准由生产者约定。
    // 推理不依赖这些元数据，也不会代替调用方生成采集时间。
    std::uint64_t sequence{0};
    std::int64_t capturedAtMs{-1};
    std::int64_t publishedAtMs{-1};

    /* 仅校验非空和基本几何；不判断像素格式、偶数约束或 NV12 字节数。 */
    constexpr bool valid() const noexcept
    {
        return !bytes.empty() && geometry.valid();
    }
};

} // namespace rk_yolo

#endif
