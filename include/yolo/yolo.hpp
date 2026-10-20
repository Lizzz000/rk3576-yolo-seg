// SPDX-License-Identifier: Apache-2.0
/**
  ******************************************************************************
  * @file    yolo.hpp
  * @brief   RK3576 通用 YOLO 分割推理接口
  ******************************************************************************
  * 使用顺序：load(模型路径, 类别数) → 设置色彩范围/NMS → detect(NV12 帧)。
  * 输入尺寸从模型读取；类别名称由调用者维护，核心只返回原始类别编号。
  * 每次读取结果时同时检查 outputTruncated()，释放资源可调用 release()，
  * 也可交给析构。一个实例串行使用，调用期间输入帧的存储必须保持有效。
  */
#ifndef RK3576_YOLO_SEG_YOLO_H
#define RK3576_YOLO_SEG_YOLO_H

#include "camera/camera_types.hpp"
#include "types/frame_types.hpp"
#include "types/result.hpp"
#include "types/vision_types.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace rk_yolo {

constexpr std::size_t YOLO_MAX_REGIONS = 64; // 单帧结果容量；不是模型类别数
constexpr std::size_t YOLO_MAX_OUTPUTS = 16; // 可查询的 RKNN 输出张量数量上限
constexpr std::size_t YOLO_MASK_CHANNELS = 32; // 当前分割模型契约的系数/prototype 通道数
constexpr int YOLO_MAX_CLASSES = 4096; // load() 允许的类别数量上限

/* 支持的检测头布局；Unknown 表示未加载。形状及输出顺序见 load()。 */
enum class YoloOutputLayout : std::uint8_t { Unknown, SplitDfl, RawSeg, PackedSeg, End2EndSeg };

/* detect() 的阶段错误；模型加载失败的具体原因从 load() 返回的 Status 读取。 */
enum class YoloError : std::uint8_t {
    None, InvalidArgument, InvalidStride, Preprocess, SetInput, Run, GetOutput, OutputData
};

/* 将源图归一化坐标映射到 letterbox 模型坐标。padding 已经处于模型
 * 像素坐标，不能与缩放因子再次相乘。sourceSize 为原图对应轴的像素尺寸。 */
inline float letterboxProjectNormalized(float normalized, int sourceSize,
                                         float scale, float padding) noexcept
{
    return normalized * static_cast<float>(sourceSize) * scale + padding;
}

/* 单目标几何结果。框、中心和底点相对原图归一化；不包含完整 mask 像素。 */
struct SegRegion final {
    // 模型原始类别编号与类别分数；不存在业务类别重映射。
    int classId{0};
    float score{0.0F};
    // mask 有效时为掩膜矩心；否则回退为检测框中心。
    float cx{0.0F}, cy{0.0F};
    // mask 底部统计点；无有效 mask 时回退为框底边中点，不代表地面坐标。
    float bottomX{0.0F}, bottomY{0.0F};
    // area/maskArea 为相对原图面积，roiMaskOverlap 为 mask 落入指定 ROI 的比例。
    float area{0.0F}, maskArea{0.0F}, roiMaskOverlap{0.0F};
    bool maskValid{false};
    // 左上角/右下角坐标，已裁剪到原图有效范围。
    float x1{0.0F}, y1{0.0F}, x2{0.0F}, y2{0.0F};
};

/* 最近一次 detect() 的阶段耗时，单位 ms；开启 setProfiling(true) 后记录。
 * inference 包含设置输入、执行与获取输出；不包含摄像头取流和 BGR→NV12。
 * 失败时可能仅有此前已完成阶段的耗时，不能据此计算完整循环 FPS。 */
struct YoloTiming final {
    double preprocessMilliseconds{0.0};
    double inferenceMilliseconds{0.0};
    double postprocessMilliseconds{0.0};
    double maskMilliseconds{0.0}; // postprocess 的子集，不可再次相加
    unsigned int maskCandidates{0}; // 本次进入 mask 统计的目标数
};

using DetectionList = FixedList<SegRegion, YOLO_MAX_REGIONS>;

/* 独占 RKNN 上下文和工作区，禁止复制；移动时转交资源所有权。 */
class YoloDetector final {
public:
    YoloDetector() noexcept;
    ~YoloDetector() noexcept;
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;
    YoloDetector(YoloDetector&&) noexcept;
    YoloDetector& operator=(YoloDetector&&) noexcept;

    /* 加载模型并检查契约，numClasses 必须与导出一致，范围 1..YOLO_MAX_CLASSES。
     * 输入为 batch=1 的 RGB，支持 NHWC/NCHW 查询布局；尺寸须能被 4 整除。
     * 32 通道 prototype 采用 NCHW，宽高为模型输入的 1/4。C 表示类别数：
     *   SplitDfl：stride 8/16/32，各为 box[1,64,H,W]、scores[1,C,H,W]、
     *             可选 gate[1,1,H,W]、coeff[1,32,H,W]，最后为 prototype。
     *   RawSeg：boxes[1,4,N]、scores[1,C,N]、coeff[1,32,N]、prototype。
     *   PackedSeg：[1,4+C+32,N]，依次为像素 cxcywh、类别分数、mask 系数。
     *   End2EndSeg：[1,N,38]，依次为像素 xyxy、分数、类别号、32 个系数。
     * 类别分数已激活，以上布局不额外包含 objectness。
     * 明确角色名称可用于重排；形状冲突且名称未知时，必须遵循完整输出顺序。
     * 同时匹配 packed/E2E 的歧义形状会被拒绝。失败原因见返回的 Status。
     * 通过参数预检查后会先释放旧模型，开始加载时将 NMS/色彩范围恢复默认值；
     * 参数预检查失败则直接返回，保留原有模型。 */
    Status load(std::string_view modelPath, int numClasses) noexcept;
    /** 释放模型与工作区，可重复调用；后续需重新 load() 才能推理。 */
    void release() noexcept;
    bool isLoaded() const noexcept;
    // 未加载时返回 0/Unknown；输入宽高的单位为模型像素。
    int classCount() const noexcept;
    int inputWidth() const noexcept;
    int inputHeight() const noexcept;
    YoloOutputLayout outputLayout() const noexcept;
    /* 配置原图归一化 ROI，需 0≤x1<x2≤1、0≤y1<y2≤1。
     * 仅用于 roiMaskOverlap 统计，不裁切输入，也不限定检测范围；默认未启用。 */
    Status setMaskRoi(float x1, float y1, float x2, float y2) noexcept;
    /** 类别内 NMS 的 IoU 门限，范围 (0,1]；建议在 load() 成功后设置。 */
    Status setNmsIou(float threshold) noexcept;
    /** 按实际 NV12 的颜色矩阵/范围设置；示例 BGR 转换使用 Bt601Limited。 */
    Status setColorSpace(CameraColorSpace colorSpace) noexcept;
    /** 启用底点局部细化，默认关闭；底点仍是图像统计量，不代表物理位置。 */
    Status setBottomRefinement(bool enabled) noexcept;
    /* 同步执行一帧推理。frame 必须包含完整连续 NV12，长度覆盖 stride 填充。
     * confidenceThreshold 范围 [0,1]；gateThreshold 非负，仅带 gate 的 DFL 使用。
     * 成功可返回空列表；失败先读取 status()，不能继续访问 value()。
     * 调用返回前输入存储不得释放或被其他线程改写。 */
    Result<DetectionList> detect(FrameView frame, float confidenceThreshold,
                                 float gateThreshold) noexcept;
    YoloError lastError() const noexcept;
    YoloTiming timing() const noexcept;
    /** true 表示结果容量不足，应将列表视为不完整，而不是“其余目标不存在”。 */
    bool outputTruncated() const noexcept;
    /** 开启阶段计时会读取单调时钟；关闭时不提供有效耗时统计。 */
    void setProfiling(bool enabled) noexcept;

    static std::string_view errorName(YoloError error) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rk_yolo

/* 下列值类型别名供内部 RKNN/RGA 适配实现使用；对外使用 rk_yolo::YoloDetector。 */
using SegRegion_t = rk_yolo::SegRegion;
using YoloTiming_t = rk_yolo::YoloTiming;
using YoloError_t = rk_yolo::YoloError;
using YoloOutputLayout_t = rk_yolo::YoloOutputLayout;
inline constexpr int YOLO_MAX_REGIONS = static_cast<int>(rk_yolo::YOLO_MAX_REGIONS);
inline constexpr int YOLO_MAX_OUTPUTS = static_cast<int>(rk_yolo::YOLO_MAX_OUTPUTS);
inline constexpr int YOLO_MASK_CHANNELS = static_cast<int>(rk_yolo::YOLO_MASK_CHANNELS);

#endif
