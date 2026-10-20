// SPDX-License-Identifier: Apache-2.0
/**
  * @file    vision_types.hpp
  * @brief   视觉结果使用的固定容量容器
  * 热路径不扩容；调用者必须处理容量不足，不能把截断列表当作完整识别结果。
  */
#ifndef RK3576_YOLO_SEG_VISION_TYPES_H
#define RK3576_YOLO_SEG_VISION_TYPES_H

#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

namespace rk_yolo
{

/* 内部存储为 std::array。迭代器只遍历 size() 个有效元素，capacity() 固定不变。 */
template <typename T, std::size_t Capacity>
class FixedList final
{
public:
    using value_type = T;
    using iterator = typename std::array<T, Capacity>::iterator;
    using const_iterator = typename std::array<T, Capacity>::const_iterator;

    constexpr std::size_t size() const noexcept { return size_; }
    constexpr std::size_t capacity() const noexcept { return Capacity; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr bool full() const noexcept { return size_ == Capacity; }
    constexpr void clear() noexcept { size_ = 0; } // 只重置有效数量，不重新分配存储

    /** 成功加入返回 true；容量已满则返回 false，已有元素不变。 */
    bool push_back(const T& value) noexcept
    {
        if (full()) return false;
        values_[size_++] = value;
        return true;
    }

    /** 移动加入版本，容量不足时不会消费传入值。 */
    bool push_back(T&& value) noexcept
    {
        if (full()) return false;
        values_[size_++] = std::move(value);
        return true;
    }

    // 下标访问不检查范围；调用者须保证 index < size()。
    constexpr T& operator[](std::size_t index) noexcept { return values_[index]; }
    constexpr const T& operator[](std::size_t index) const noexcept { return values_[index]; }
    constexpr iterator begin() noexcept { return values_.begin(); }
    constexpr iterator end() noexcept { return values_.begin() + static_cast<std::ptrdiff_t>(size_); }
    constexpr const_iterator begin() const noexcept { return values_.begin(); }
    constexpr const_iterator end() const noexcept { return values_.begin() + static_cast<std::ptrdiff_t>(size_); }
    constexpr T* data() noexcept { return values_.data(); }
    constexpr const T* data() const noexcept { return values_.data(); }

private:
    std::array<T, Capacity> values_{};
    std::size_t size_{0};
};

} // namespace rk_yolo

#endif
