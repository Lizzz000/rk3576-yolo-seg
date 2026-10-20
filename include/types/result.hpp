// SPDX-License-Identifier: Apache-2.0
/**
  * @file    result.hpp
  * @brief   状态码与带值结果的统一返回类型
  * 调用方式：先检查 if (result)，成功后再读 value()；失败时读 status()。
  */
#ifndef RK3576_YOLO_SEG_RESULT_H
#define RK3576_YOLO_SEG_RESULT_H

#include <string>
#include <string_view>
#include <optional>
#include <utility>

namespace rk_yolo
{

/* 统一错误分类；None 表示成功，其余值供调用者选择报错、重试或退出。 */
enum class ErrorCode
{
    None = 0,
    InvalidArgument,
    NotInitialized,
    AlreadyRunning,
    Busy,
    Io,
    Timeout,
    Unsupported,
    OutOfMemory,
    Thread,
    ResourceUnavailable,
    Hardware,
    Internal
};

/** 返回错误码的固定文本视图，不分配字符串。 */
constexpr std::string_view ErrorCodeName(ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::None: return "none";
    case ErrorCode::InvalidArgument: return "invalid_argument";
    case ErrorCode::NotInitialized: return "not_initialized";
    case ErrorCode::AlreadyRunning: return "already_running";
    case ErrorCode::Busy: return "busy";
    case ErrorCode::Io: return "io";
    case ErrorCode::Timeout: return "timeout";
    case ErrorCode::Unsupported: return "unsupported";
    case ErrorCode::OutOfMemory: return "out_of_memory";
    case ErrorCode::Thread: return "thread";
    case ErrorCode::ResourceUnavailable: return "resource_unavailable";
    case ErrorCode::Hardware: return "hardware";
    case ErrorCode::Internal: return "internal";
    }
    return "unknown";
}

struct Status
{
    // bool/ok() 只按 code 判断；message 为可选的具体原因。
    ErrorCode code{ErrorCode::None};
    std::string message{};

    Status() = default;
    explicit Status(ErrorCode value) noexcept : code(value) {}
    Status(ErrorCode value, std::string text)
        : code(value), message(std::move(text)) {}

    constexpr explicit operator bool() const noexcept
    {
        return code == ErrorCode::None;
    }

    constexpr bool ok() const noexcept
    {
        return static_cast<bool>(*this);
    }

    static Status success() noexcept { return {}; }
    static Status failure(ErrorCode value) noexcept { return Status{value}; }
    static Status failure(ErrorCode value, std::string text)
    {
        return Status{value, std::move(text)};
    }
};

/* 带值结果：成功路径使用 success(value)，失败路径使用 failure(...)。
 * value() 不做空值检查，必须先检查结果状态；移动读取会转交值内的资源。 */
template <typename T>
class Result final
{
public:
    Result(Status status) noexcept : status_(std::move(status)) {}
    Result(T value) : value_(std::move(value)), status_{} {}

    static Result success(T value) { return Result(std::move(value)); }
    static Result failure(ErrorCode code) { return Result(Status::failure(code)); }
    static Result failure(ErrorCode code, std::string text)
    {
        return Result(Status::failure(code, std::move(text)));
    }

    explicit operator bool() const noexcept { return status_.ok(); }
    bool ok() const noexcept { return status_.ok(); }
    const Status& status() const noexcept { return status_; }
    T& value() & noexcept { return *value_; }
    const T& value() const& noexcept { return *value_; }
    T&& value() && noexcept { return std::move(*value_); }

private:
    std::optional<T> value_{};
    Status status_{};
};

/* 无返回值操作的特化，只保存状态，默认构造表示成功。 */
template <>
class Result<void> final
{
public:
    Result() noexcept = default;
    Result(Status status) noexcept : status_(std::move(status)) {}
    explicit operator bool() const noexcept { return status_.ok(); }
    bool ok() const noexcept { return status_.ok(); }
    const Status& status() const noexcept { return status_; }

private:
    Status status_{};
};

} // namespace rk_yolo

#endif
