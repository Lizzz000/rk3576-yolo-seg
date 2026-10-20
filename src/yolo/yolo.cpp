// SPDX-License-Identifier: Apache-2.0
/**
  ******************************************************************************
  * @file    src/yolo/yolo.cpp
  * @brief   RK3576 YOLO 实例分割推理、输出布局校验与几何后处理
  ******************************************************************************
  * 使用流程：先 load(modelPath, numClasses)，再设置颜色空间、NMS 等选项，
  *           最后逐帧调用 detect()；模型文件和类别含义由调用方提供。
  * 加载阶段：检查输入/输出尺寸、数据类型和量化属性，匹配分尺度 DFL、
  *           raw split、packed 或 E2E 布局，并准备可复用的后处理工作区。
  * 每帧流程：NV12 经 RGA 等比例缩放、补边并转 RGB → RKNN 推理 →
  *           解码候选框 → 同类 NMS → 掩膜矩心、面积、底点与 ROI 重叠率。
  * 坐标约定：解码与 NMS 使用模型输入像素；对外结果映射回原图并归一化。
  * 资源边界：输入图像由调用方持有；RKNN 输出在本次推理内释放，返回值
  *           只包含独立的几何结果。该核心不负责图像采集、显示或控制设备。
  ******************************************************************************
  */
#include "yolo/yolo.hpp"
#include "yolo/mask_projection.hpp"
#include "yolo/input_validation.hpp"

#include <algorithm>
#include <stddef.h>
#include <rknn_api.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <rga/im2d.h>
#pragma GCC diagnostic pop

#include <math.h>
#include <memory>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace {

using YoloError_t = rk_yolo::YoloError;
using YoloOutputLayout_t = rk_yolo::YoloOutputLayout;
using CameraColorSpace_t = rk_yolo::CameraColorSpace;
using SegRegion_t = rk_yolo::SegRegion;
using YoloTiming_t = rk_yolo::YoloTiming;

/**
 * @brief 单个检测器独占的模型状态与可复用缓冲区。
 * @note  输出角色索引在加载时确定；推理只按这些索引读取，不按类别名分支。
 *        letterbox 与 mask 缓冲区随模型释放，同一实例的调用由调用方串行组织。
 */
struct YoloStorage {
    rknn_context ctx{};
    int inW{0}, inH{0}, nOut{0}, nScale{0}, numClasses{0};
    int gridH[3]{}, gridW[3]{}, idxBox[3]{}, idxClass[3]{}, idxGate[3]{}, idxMask[3]{};
    int idxProto{-1}, idxDetection{-1}, idxRawBox{-1}, idxRawClass{-1}, idxRawMask{-1};
    int nPredictions{0}; YoloOutputLayout_t outputLayout{YoloOutputLayout_t::Unknown};
    int maskW{0}, maskH{0};
    rknn_tensor_attr outputAttrs[rk_yolo::YOLO_MAX_OUTPUTS]{};
    int useNativeInt8{0}, useNativeFp16{0};
    std::vector<std::uint8_t> letterboxStorage;
    std::vector<std::uint32_t> maskPoints;
    std::vector<unsigned int> maskRowCounts;
    std::vector<std::int32_t> maskRowLogits;
    void* workspace{nullptr};
    YoloError_t lastError{YoloError_t::None};
    int lastOutputTruncated{0};
    float nmsIouThreshold{0.45F};
    CameraColorSpace_t colorSpace{CameraColorSpace_t::Bt601Full};
    int profilingEnabled{0};
    YoloTiming_t lastTiming{};
    int maskRoiEnabled{0};
    float maskRoiX1{0.0F}, maskRoiY1{0.0F}, maskRoiX2{1.0F}, maskRoiY2{1.0F};
    int bottomRefineEnabled{0};
    int letterboxGeometryValid{0};
    int letterboxWidth{0}, letterboxHeight{0}, letterboxPadX{0}, letterboxPadY{0};
    int letterboxResizedW{0}, letterboxResizedH{0};
};
using Yolo_t = YoloStorage;

constexpr auto YOLO_OUTPUT_SPLIT_DFL = YoloOutputLayout_t::SplitDfl;
constexpr auto YOLO_OUTPUT_PACKED_SEG = YoloOutputLayout_t::PackedSeg;
constexpr auto YOLO_OUTPUT_END2END_SEG = YoloOutputLayout_t::End2EndSeg;
constexpr auto YOLO_OUTPUT_RAW_SEG = YoloOutputLayout_t::RawSeg;
constexpr auto YOLO_ERROR_NONE = YoloError_t::None;
constexpr auto YOLO_ERROR_INVALID_ARGUMENT = YoloError_t::InvalidArgument;
constexpr auto YOLO_ERROR_INVALID_STRIDE = YoloError_t::InvalidStride;
constexpr auto YOLO_ERROR_PREPROCESS = YoloError_t::Preprocess;
constexpr auto YOLO_ERROR_SET_INPUT = YoloError_t::SetInput;
constexpr auto YOLO_ERROR_RUN = YoloError_t::Run;
constexpr auto YOLO_ERROR_GET_OUTPUT = YoloError_t::GetOutput;
constexpr auto YOLO_ERROR_OUTPUT_DATA = YoloError_t::OutputData;

static void Yolo_Release(Yolo_t* yolo);

/* ==================== 内部工具 ==================== */

/* 负数表示本帧推理失败；成功但无目标返回 0，二者不可混作空检测。 */
static int detectError(Yolo_t* yolo, YoloError_t error)
{
    if (yolo != nullptr)
        yolo->lastError = error;
    return -1;
}

/* 性能统计使用单调时钟；关闭 profiling 时热路径不会调用此函数。 */
static double profileMilliseconds(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

/**
 * @brief 读取 NPU 核选择选项，并给出用于日志的名称。
 * @note  这里只生成请求；实际是否接受由 RKNN 决定，失败时保留运行时调度。
 */
static rknn_core_mask requestedCoreMask(const char** name)
{
    const char* requested = getenv("RK_YOLO_RKNN_CORE");
    if (requested == nullptr || strcmp(requested, "core0_1") == 0)
    {
        *name = "CORE_0_1";
        return RKNN_NPU_CORE_0_1;
    }
    if (strcmp(requested, "auto") == 0)
    {
        *name = "AUTO";
        return RKNN_NPU_CORE_AUTO;
    }
    if (strcmp(requested, "core0") == 0)
    {
        *name = "CORE_0";
        return RKNN_NPU_CORE_0;
    }
    if (strcmp(requested, "core1") == 0)
    {
        *name = "CORE_1";
        return RKNN_NPU_CORE_1;
    }
    if (strcmp(requested, "all") == 0)
    {
        *name = "ALL";
        return RKNN_NPU_CORE_ALL;
    }
    printf("[YOLO] 未知 RK_YOLO_RKNN_CORE=%s，使用 CORE_0_1\n", requested);
    *name = "CORE_0_1";
    return RKNN_NPU_CORE_0_1;
}

/**
 * @brief 解码四条边的 DFL 分布，返回相对网格中心的距离。
 * @note  来源参考 Rockchip rknn_model_zoo 的 postprocess.cc / compute_dfl，
 *        已调整为先减最大值的稳定 softmax；保留来源，采用 Apache-2.0。
 * @note  当前导出契约为每边 16 个 logits；先减最大值再做 softmax，
 *        得到的距离以网格单元计，调用处再乘 stride 转为模型像素。
 */
static void computeDfl(const float* tensor, int dflLen, float* box)
{
    float expT[16];
    for (int b = 0; b < 4; ++b)
    {
        float maxLogit = tensor[b * dflLen];
        for (int i = 1; i < dflLen; ++i)
            maxLogit = fmaxf(maxLogit, tensor[i + b * dflLen]);
        float expSum = 0.0f, acc = 0.0f;
        for (int i = 0; i < dflLen; ++i)
        {
            expT[i] = expf(tensor[i + b * dflLen] - maxLogit);
            expSum += expT[i];
        }
        for (int i = 0; i < dflLen; ++i)
            acc += expT[i] / expSum * (float)i;
        box[b] = acc;
    }
}

/* NMS 使用同一模型像素坐标系的框计算 IoU；退化面积返回 0。 */
static float boxIou(float ax1, float ay1, float ax2, float ay2,
                    float bx1, float by1, float bx2, float by2)
{
    const float ix1 = fmaxf(ax1, bx1), iy1 = fmaxf(ay1, by1);
    const float ix2 = fminf(ax2, bx2), iy2 = fminf(ay2, by2);
    const float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
    const float uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/* 候选框尚未去除 letterbox 补边，四个边界均以模型输入像素计。 */
typedef struct
{
    float x1, y1, x2, y2;
    float score;
    int cls;
    int scaleIndex;   /* DFL 的尺度索引；其他布局为 -1 */
    int gridOffset;   /* DFL 的网格偏移，或其他布局的预测条目索引 */
} Cand;

/* 加载模型时准备输出描述和候选空间，后续帧复用这些容器。
   DFL 的容量取三尺度网格数之和；其他布局取预测条目数 N。
   DFL/raw/packed 每个网格或条目选最高分类分数，E2E 沿用条目中的类别与分数。
   最终返回数量另受 64 个结果上限约束。 */
struct YoloWorkspace
{
    std::vector<rknn_output> outputs;
    std::vector<Cand> candidates;
    size_t candidateCapacity = 0;
};

/* 按 IEEE-754 位域将原生 FP16 还原成 FP32，覆盖零、非规格化数与特殊值；
   这里转换存储格式，不为后处理自行降低精度。 */
static float halfToFloat(uint16_t half)
{
    const uint32_t sign = ((uint32_t)half & 0x8000U) << 16U;
    uint32_t exponent = ((uint32_t)half >> 10U) & 0x1FU;
    uint32_t mantissa = (uint32_t)half & 0x03FFU;
    uint32_t bits;
    if (exponent == 0U)
    {
        if (mantissa == 0U)
            bits = sign;
        else
        {
            int unbiasedExponent = -14;
            while ((mantissa & 0x0400U) == 0U)
            {
                mantissa <<= 1U;
                --unbiasedExponent;
            }
            mantissa &= 0x03FFU;
            bits = sign | ((uint32_t)(unbiasedExponent + 127) << 23U)
                 | (mantissa << 13U);
        }
    }
    else if (exponent == 0x1FU)
        bits = sign | 0x7F800000U | (mantissa << 13U);
    else
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* 按加载时确定的统一输出模式读取元素：INT8 用 (q-zp)*scale 反量化，
   FP16 按需转换；其余情况读取 RKNN 已转换的 FP32。
   此处不再猜测类型，避免混合类型输出被同一种指针错误解释。 */
static inline float outputValue(const Yolo_t* yolo, const rknn_output* outs,
                                int outputIndex, size_t elementIndex)
{
    if (yolo->useNativeInt8)
    {
        const int8_t* values = (const int8_t*)outs[outputIndex].buf;
        const rknn_tensor_attr* attr = &yolo->outputAttrs[outputIndex];
        return ((float)values[elementIndex] - (float)attr->zp) * attr->scale;
    }
    if (yolo->useNativeFp16)
        return halfToFloat(((const uint16_t*)outs[outputIndex].buf)[elementIndex]);
    return ((const float*)outs[outputIndex].buf)[elementIndex];
}

/* 将调用方提供的 NV12 色彩标准传给 RGA；颜色范围会影响模型实际输入。 */
static int rgaColorSpaceMode(CameraColorSpace_t colorSpace)
{
    switch (colorSpace)
    {
        case CAMERA_COLOR_BT601_LIMITED: return IM_YUV_TO_RGB_BT601_LIMIT;
        case CAMERA_COLOR_BT709_LIMITED: return IM_YUV_TO_RGB_BT709_LIMIT;
        case CAMERA_COLOR_BT601_FULL:
        default: return IM_YUV_TO_RGB_BT601_FULL;
    }
}

/**
 * @brief 将候选的 32 个 mask 系数投影到 prototype，统计掩膜几何。
 * @param scale 本帧 letterbox 的缩放比例。
 * @param padX 水平方向补边，单位为模型像素。
 * @param padY 竖直方向补边，单位为模型像素。
 * @note  mask>0.5 等价于 sigmoid 前的 logit>0；只统计框内且能映射到
 *        原图的 prototype 像素。中心、底点和面积均按原图归一化。
 *        返回 0 表示没有有效掩膜，保留调用方传入的框几何作为后备值；
 *        ROI 重叠率为有效掩膜落入 ROI 的比例，不是框的交并比。
 */
static int computeMaskCentroid(Yolo_t* yolo, const Cand* cand,
                               const rknn_output* outs,
                               float scale, float padX, float padY,
                               int width, int height,
                               float fallbackCx, float fallbackCy, float fallbackArea,
                               float fallbackBottomX, float fallbackBottomY,
                               float* outCx, float* outCy, float* outArea,
                               float* outBottomX, float* outBottomY,
                               float* outRoiOverlap)
{
    *outCx = fallbackCx;
    *outCy = fallbackCy;
    *outArea = fallbackArea;
    *outBottomX = fallbackBottomX;
    *outBottomY = fallbackBottomY;
    *outRoiOverlap = 0.0f;

    if (cand->cls < 0 || cand->cls >= yolo->numClasses || yolo->idxProto < 0)
        return 0;

    /* 不同布局的系数存放位置不同：E2E 按条目排列，raw/DFL 按通道排列，
       packed 的系数还需越过 4 个框通道和 numClasses 个分数通道。 */
    int maskIndex = -1;
    int gridLen = 0;
    if (yolo->outputLayout == YOLO_OUTPUT_END2END_SEG)
        maskIndex = yolo->idxDetection;
    else if (yolo->outputLayout == YOLO_OUTPUT_RAW_SEG
             || yolo->outputLayout == YOLO_OUTPUT_PACKED_SEG)
    {
        maskIndex = yolo->outputLayout == YOLO_OUTPUT_PACKED_SEG
            ? yolo->idxDetection : yolo->idxRawMask;
        gridLen = yolo->nPredictions;
    }
    else if (cand->scaleIndex >= 0 && cand->scaleIndex < yolo->nScale)
    {
        maskIndex = yolo->idxMask[cand->scaleIndex];
        gridLen = yolo->gridW[cand->scaleIndex]
                * yolo->gridH[cand->scaleIndex];
    }
    if (maskIndex < 0 || outs[maskIndex].buf == nullptr || outs[yolo->idxProto].buf == nullptr)
        return 0;

    float coeff[YOLO_MASK_CHANNELS];
    int16_t quantizedCoeff[YOLO_MASK_CHANNELS];
    const int8_t* coeffTensor = (const int8_t*)outs[maskIndex].buf;
    const int8_t* protoTensor = (const int8_t*)outs[yolo->idxProto].buf;
    const int coeffZeroPoint = yolo->outputAttrs[maskIndex].zp;
    const int protoZeroPoint = yolo->outputAttrs[yolo->idxProto].zp;
    for (int k = 0; k < YOLO_MASK_CHANNELS; ++k)
    {
        const size_t index = yolo->outputLayout == YOLO_OUTPUT_END2END_SEG
            ? (size_t)cand->gridOffset * (size_t)(6 + YOLO_MASK_CHANNELS)
              + (size_t)(6 + k)
            : (size_t)(k + (yolo->outputLayout == YOLO_OUTPUT_PACKED_SEG ? 4 + yolo->numClasses : 0))
                * (size_t)gridLen + (size_t)cand->gridOffset;
        if (yolo->useNativeInt8)
            quantizedCoeff[k] = (int16_t)((int)coeffTensor[index]
                                          - coeffZeroPoint);
        else
            coeff[k] = outputValue(yolo, outs, maskIndex, index);
    }

    /* 将模型像素框裁到 prototype 网格后再转整数，避免越界读取。 */
    const float maskScaleX = (float)yolo->maskW / (float)yolo->inW;
    const float maskScaleY = (float)yolo->maskH / (float)yolo->inH;
    const int xBegin = (int)floorf(fmaxf(0.0F, fminf((float)yolo->maskW, cand->x1 * maskScaleX)));
    const int yBegin = (int)floorf(fmaxf(0.0F, fminf((float)yolo->maskH, cand->y1 * maskScaleY)));
    const int xEnd = (int)ceilf(fmaxf(0.0F, fminf((float)yolo->maskW, cand->x2 * maskScaleX)));
    const int yEnd = (int)ceilf(fmaxf(0.0F, fminf((float)yolo->maskH, cand->y2 * maskScaleY)));
    if (xEnd <= xBegin || yEnd <= yBegin)
        return 0;

    const float widthF = (float)width;
    const float heightF = (float)height;
    const float modelPerMaskX = (float)yolo->inW / (float)yolo->maskW;
    const float modelPerMaskY = (float)yolo->inH / (float)yolo->maskH;
    float sumX = 0.0f;
    float sumY = 0.0f;
    int count = 0;
    int roiCount = 0;
    yolo->maskPoints.clear();
    std::fill(yolo->maskRowCounts.begin(), yolo->maskRowCounts.end(), 0U);

    const size_t plane = (size_t)yolo->maskW * (size_t)yolo->maskH;
    for (int y = yBegin; y < yEnd; ++y)
    {
        const float modelY = ((float)y + 0.5f) * modelPerMaskY;
        const float normY = (modelY - padY) / scale / heightF;
        if (normY < 0.0f || normY > 1.0f)
            continue;
        /* INT8 的量化 scale 均为正，整数点积的正负与实数 logit 一致。
           按通道连续投影一行，再统计该行像素，不改变 32 通道或阈值。 */
        if (yolo->useNativeInt8)
            MaskProjection::int8Row(protoTensor, quantizedCoeff, YOLO_MASK_CHANNELS,
                plane, (size_t)y * yolo->maskW + xBegin, (size_t)(xEnd - xBegin),
                protoZeroPoint, yolo->maskRowLogits.data());
        for (int x = xBegin; x < xEnd; ++x)
        {
            const size_t pixel = (size_t)y * (size_t)yolo->maskW + (size_t)x;
            if (yolo->useNativeInt8)
            {
                // 与原整数点积同值；系数和proto量化scale均为正。
                if (yolo->maskRowLogits[(size_t)(x - xBegin)] <= 0)
                    continue;
            }
            else
            {
                float logit = 0.0f;
                for (int k = 0; k < YOLO_MASK_CHANNELS; ++k)
                    logit += coeff[k] * outputValue(
                        yolo, outs, yolo->idxProto, (size_t)k * plane + pixel);
                if (!isfinite(logit) || logit <= 0.0f)
                    continue;
            }

            const float modelX = ((float)x + 0.5f) * modelPerMaskX;
            const float normX = (modelX - padX) / scale / widthF;
            if (normX < 0.0f || normX > 1.0f)
                continue;
            sumX += normX;
            sumY += normY;
            ++count;
            /* 打包原型坐标，第二遍只扫描底部窄带，避免为确定分位数
               重复执行 32 通道掩膜点积。容量已在加载时预留。 */
            if (yolo->maskPoints.size() < yolo->maskPoints.capacity())
                yolo->maskPoints.push_back(
                    (uint32_t)y * (uint32_t)yolo->maskW + (uint32_t)x);
            if ((size_t)y < yolo->maskRowCounts.size())
                ++yolo->maskRowCounts[(size_t)y];
            if (yolo->maskRoiEnabled
                && normX >= yolo->maskRoiX1 && normX <= yolo->maskRoiX2
                && normY >= yolo->maskRoiY1 && normY <= yolo->maskRoiY2)
                ++roiCount;
        }
    }

    /* 没有正掩膜像素时不伪造面积或 ROI 证据，交由 maskValid 告知调用方。 */
    if (count <= 0)
        return 0;

    *outCx = fminf(1.0f, fmaxf(0.0f, sumX / (float)count));
    *outCy = fminf(1.0f, fmaxf(0.0f, sumY / (float)count));
    const float cellArea = (modelPerMaskX / scale / widthF)
                         * (modelPerMaskY / scale / heightF);
    *outArea = fminf(1.0f, (float)count * cellArea);
    /* 底点纵坐标取有效掩膜像素纵坐标的 97% 分位，再取底部窄带的横向中位数；
       它描述图像中的掩膜底部，不表示实物距离或接触地面的测量值。 */
    if (!yolo->maskPoints.empty())
    {
        const size_t quantileRank = std::max<size_t>(
            1U, (size_t)std::ceil(0.97 * (double)yolo->maskPoints.size()));
        size_t cumulativeRows = 0U;
        int bottomRow = yBegin;
        for (int row = yBegin; row < yEnd; ++row)
        {
            cumulativeRows += yolo->maskRowCounts[(size_t)row];
            if (cumulativeRows >= quantileRank)
            {
                bottomRow = row;
                break;
            }
        }
        const int boxHeightRows = std::max(1, yEnd - yBegin);
        const int bandRows = std::max(3,
            (int)std::ceil(0.05 * (double)boxHeightRows));
        const int bandStart = std::max(yBegin, bottomRow - bandRows + 1);
        unsigned int bottomHistogram[256] = {};
        int bottomCount = 0;
        for (const uint32_t packed : yolo->maskPoints)
        {
            const int y = (int)(packed / (uint32_t)yolo->maskW);
            if (y < bandStart || y > bottomRow)
                continue;
            const int x = (int)(packed % (uint32_t)yolo->maskW);
            const float modelX = ((float)x + 0.5f) * modelPerMaskX;
            const float normX = (modelX - padX) / scale / widthF;
            if (normX < 0.0f || normX > 1.0f)
                continue;
            const int bin = std::max(0, std::min(255,
                (int)std::floor(normX * 256.0f)));
            ++bottomHistogram[bin];
            ++bottomCount;
        }
        if (bottomCount > 0)
        {
            unsigned int cumulative = 0U;
            const unsigned int medianRank = (unsigned int)(bottomCount + 1) / 2U;
            int medianBin = 0;
            for (; medianBin < 255; ++medianBin)
            {
                cumulative += bottomHistogram[medianBin];
                if (cumulative >= medianRank)
                    break;
            }
            *outBottomX = ((float)medianBin + 0.5f) / 256.0f;
            float refinedBottomRow = (float)bottomRow + 0.5f;
            if (yolo->bottomRefineEnabled)
            {
                /* 可选底边细化：按横向分箱统计底部窄带内的最下有效原型行。
                   以中位数/MAD 筛选异常值，再对保留行中心取均值；
                   行中心含 0.5 偏移，映射为连续坐标近似，默认关闭。 */
                float columnRows[256] = {};
                float deviations[256] = {};
                int columnCount = 0;
                int maxRows[256];
                for (int& row : maxRows)
                    row = -1;
                for (const uint32_t packed : yolo->maskPoints)
                {
                    const int y = (int)(packed / (uint32_t)yolo->maskW);
                    if (y < bandStart || y > bottomRow)
                        continue;
                    const int x = (int)(packed % (uint32_t)yolo->maskW);
                    const int bin = std::max(0, std::min(255,
                        (int)std::floor((((float)x + 0.5f)
                            * modelPerMaskX - padX) / scale / widthF * 256.0f)));
                    maxRows[bin] = std::max(maxRows[bin], y);
                }
                for (int bin = 0; bin < 256; ++bin)
                    if (maxRows[bin] >= 0)
                        if (columnCount < 256)
                            columnRows[columnCount++] = (float)maxRows[bin] + 0.5f;
                if (columnCount >= 3)
                {
                    std::sort(columnRows, columnRows + columnCount);
                    const float median = columnRows[columnCount / 2];
                    for (int i = 0; i < columnCount; ++i)
                        deviations[i] = fabsf(columnRows[i] - median);
                    std::sort(deviations, deviations + columnCount);
                    const float mad = deviations[columnCount / 2];
                    const float limit = 2.5f * mad + 1.0f;
                    float sum = 0.0f;
                    int countRows = 0;
                    for (int i = 0; i < columnCount; ++i)
                        if (fabsf(columnRows[i] - median) <= limit)
                        {
                            sum += columnRows[i];
                            ++countRows;
                        }
                    if (countRows >= 3)
                        refinedBottomRow = sum / (float)countRows;
                }
            }
            const float modelBottomY = refinedBottomRow * modelPerMaskY;
            *outBottomY = fmaxf(0.0f, fminf(1.0f,
                (modelBottomY - padY) / scale / heightF));
        }
    }
    *outRoiOverlap = (float)roiCount / (float)count;
    return 1;
}

/**
 * @brief 用 RGA 将含 stride 补齐的 NV12 转为模型尺寸的 RGB letterbox。
 * @note  保持宽高比，空白区填 RGB 114；scale 是缩放比例，padX/padY
 *        已是模型像素，后续坐标逆变换必须先减补边再除缩放比例。
 *        目标描述覆盖完整缓冲区，缩放区域仅由 destinationRect 指定。
 *        输入几何变化时重新清理补边，固定几何时复用同一 RGB 缓冲区。
 *        RGA 错误立即返回失败，不把未完成的输入继续交给 RKNN。
 */
static int letterbox(Yolo_t* yolo, const uint8_t* nv12, int width, int height,
                     int strideW, int strideH,
                     float* outScale, float* outPadX, float* outPadY)
{
    const float widthF = (float)width;
    const float heightF = (float)height;
    const float scale = fminf((float)yolo->inW / widthF,
                              (float)yolo->inH / heightF);
    const int scaledW = (int)((float)width * scale);
    const int scaledH = (int)((float)height * scale);
    const int newW = scaledW > 0 ? scaledW : 1;
    const int newH = scaledH > 0 ? scaledH : 1;
    const int padXi = (yolo->inW - newW) / 2;
    const int padYi = (yolo->inH - newH) / 2;
    *outScale = scale;
    *outPadX = (float)padXi;
    *outPadY = (float)padYi;

    const bool geometryChanged = !yolo->letterboxGeometryValid
        || yolo->letterboxWidth != width || yolo->letterboxHeight != height
        || yolo->letterboxPadX != padXi || yolo->letterboxPadY != padYi
        || yolo->letterboxResizedW != newW || yolo->letterboxResizedH != newH;
    if (geometryChanged)
    {
        /* RGA 只覆盖 drect，几何变化时必须清除完整目标缓冲区，避免
           上一张不同宽高比图像的 padding 残留影响推理。 */
        std::fill(yolo->letterboxStorage.begin(),
                  yolo->letterboxStorage.end(), 114U);
        yolo->letterboxGeometryValid = 1;
        yolo->letterboxWidth = width;
        yolo->letterboxHeight = height;
        yolo->letterboxPadX = padXi;
        yolo->letterboxPadY = padYi;
        yolo->letterboxResizedW = newW;
        yolo->letterboxResizedH = newH;
    }

    rga_buffer_t source = wrapbuffer_virtualaddr_t(
        (void*)nv12, width, height, strideW, strideH,
        RK_FORMAT_YCbCr_420_SP);
    source.color_space_mode = rgaColorSpaceMode(yolo->colorSpace);
    rga_buffer_t resized = wrapbuffer_virtualaddr_t(
        yolo->letterboxStorage.data(), yolo->inW, yolo->inH,
        yolo->inW, yolo->inH, RK_FORMAT_RGB_888);
    const im_rect sourceRect{0, 0, width, height};
    const im_rect destinationRect{padXi, padYi, newW, newH};
    const im_rect patternRect{0, 0, 0, 0};
    rga_buffer_t pattern{};
    im_opt_t options{};
    options.interp = IM_INTERP(IM_INTERP_LINEAR, IM_INTERP_LINEAR);
    const IM_STATUS resizeStatus = improcess(
        source, resized, pattern, sourceRect, destinationRect, patternRect,
        -1, nullptr, &options, IM_SYNC);
    if (resizeStatus != IM_STATUS_SUCCESS)
    {
        printf("[YOLO] RGA 预处理失败: %s\n", imStrError_t(resizeStatus));
        return 0;
    }

    return 1;
}

/* ==================== 内部状态配置 ==================== */

int Yolo_IsLoaded(const Yolo_t* yolo)
{
    return yolo != nullptr && yolo->ctx != 0;
}

/* ROI 使用原图归一化边界，只影响掩膜重叠率统计，不裁剪模型输入。 */
int Yolo_SetMaskRoi(Yolo_t* yolo, float x1, float y1, float x2, float y2)
{
    if (yolo == nullptr || !isfinite(x1) || !isfinite(y1)
        || !isfinite(x2) || !isfinite(y2)
        || x1 < 0.0f || y1 < 0.0f || x2 > 1.0f || y2 > 1.0f
        || x1 >= x2 || y1 >= y2)
        return 0;
    yolo->maskRoiX1 = x1;
    yolo->maskRoiY1 = y1;
    yolo->maskRoiX2 = x2;
    yolo->maskRoiY2 = y2;
    yolo->maskRoiEnabled = 1;
    return 1;
}

int Yolo_SetNmsIou(Yolo_t* yolo, float threshold)
{
    if (yolo == nullptr || !isfinite(threshold)
        || threshold <= 0.0f || threshold > 1.0f)
        return 0;
    yolo->nmsIouThreshold = threshold;
    return 1;
}

int Yolo_SetColorSpace(Yolo_t* yolo, CameraColorSpace_t colorSpace)
{
    if (yolo == nullptr || colorSpace < CAMERA_COLOR_BT601_FULL
        || colorSpace > CAMERA_COLOR_BT709_LIMITED)
        return 0;
    yolo->colorSpace = colorSpace;
    return 1;
}

int Yolo_SetBottomRefinement(Yolo_t* yolo, int enabled)
{
    if (yolo == nullptr || (enabled != 0 && enabled != 1))
        return 0;
    yolo->bottomRefineEnabled = enabled;
    return 1;
}


/* 角色表示张量的用途，与类别 ID 或类别名称无关。 */
enum class TensorRole { Unknown, Box, Scores, Gate, Coeff, Proto };

/**
 * @brief 从明确的导出名称识别角色，允许名称带尺度后缀。
 * @note  未识别的名称不直接报错，仍须由形状或约定顺序完成无歧义匹配；
 *        已识别的名称则必须与对应角色的实际形状一致。
 */
static TensorRole tensorRole(const rknn_tensor_attr& attr) noexcept
{
    std::string_view name(attr.name);
    for (const std::string_view suffix :
         {"_s8", "_s16", "_s32", "_stride8", "_stride16", "_stride32", "_8", "_16", "_32"})
        if (name.size() >= suffix.size()
            && name.substr(name.size() - suffix.size()) == suffix) {
            name.remove_suffix(suffix.size());
            break;
        }
    if (name == "boxes" || name == "box" || name == "bbox" || name == "dfl" || name == "box_dfl")
        return TensorRole::Box;
    if (name == "class_scores" || name == "scores" || name == "classes" || name == "cls")
        return TensorRole::Scores;
    if (name == "gate" || name == "score_sum" || name == "score_sums")
        return TensorRole::Gate;
    if (name == "mask_coefficients" || name == "mask_coeffs" || name == "mask_coeff"
        || name == "coeff" || name == "coeffs")
        return TensorRole::Coeff;
    if (name == "proto" || name == "protos" || name == "mask_proto" || name == "prototype")
        return TensorRole::Proto;
    return TensorRole::Unknown;
}

static std::size_t tensorElementBytes(rknn_tensor_type type) noexcept
{
    switch (type) {
    case RKNN_TENSOR_INT8:
    case RKNN_TENSOR_UINT8: return 1U;
    case RKNN_TENSOR_FLOAT16: return 2U;
    case RKNN_TENSOR_FLOAT32: return 4U;
    default: return 0U;
    }
}

/**
 * @brief 检查张量元数据，使后续元素索引与字节数计算有明确边界。
 * @note  维度乘积必须匹配 n_elems，且能容纳 FP32 回退输出；整数输出
 *        必须有有效的仿射量化参数。这里验证存储契约，不验证模型准确率。
 */
static rk_yolo::Status validateTensor(const rknn_tensor_attr& attr, bool input)
{
    using rk_yolo::ErrorCode;
    using rk_yolo::Status;
    const std::string prefix = input ? "input tensor: " : "output tensor " + std::to_string(attr.index) + ": ";
    if (attr.n_dims == 0 || attr.n_dims > sizeof(attr.dims) / sizeof(attr.dims[0]))
        return Status::failure(ErrorCode::Unsupported, prefix + "invalid rank");
    std::size_t elements = 1U;
    for (std::uint32_t i = 0; i < attr.n_dims; ++i) {
        if (attr.dims[i] == 0U || attr.dims[i] > static_cast<std::uint32_t>(INT_MAX)
            || elements > std::numeric_limits<std::uint32_t>::max() / attr.dims[i])
            return Status::failure(ErrorCode::Unsupported, prefix + "invalid or overflowing dimensions");
        elements *= attr.dims[i];
    }
    // 为 RKNN 的 FP32 回退输出预留字节范围，其长度字段为 uint32_t。
    if (elements != attr.n_elems || elements > std::numeric_limits<std::uint32_t>::max() / 4U)
        return Status::failure(ErrorCode::Unsupported, prefix + "element count does not match dimensions or is too large");
    const std::size_t elementBytes = tensorElementBytes(attr.type);
    if (elementBytes == 0U)
        return Status::failure(ErrorCode::Unsupported, prefix + "only INT8, UINT8, FP16 and FP32 are supported");
    if (attr.size < elements * elementBytes)
        return Status::failure(ErrorCode::Unsupported, prefix + "reported byte size is smaller than the tensor");
    if (std::memchr(attr.name, '\0', sizeof(attr.name)) == nullptr)
        return Status::failure(ErrorCode::Unsupported, prefix + "unterminated tensor name");
    const bool integer = attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8;
    if (integer) {
        if (input && attr.qnt_type == RKNN_TENSOR_QNT_NONE)
            return Status::success();
        const int minimum = attr.type == RKNN_TENSOR_INT8 ? -128 : 0;
        const int maximum = attr.type == RKNN_TENSOR_INT8 ? 127 : 255;
        if (attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
            || !std::isfinite(attr.scale) || attr.scale <= 0.0F
            || attr.zp < minimum || attr.zp > maximum)
            return Status::failure(ErrorCode::Unsupported, prefix + "invalid affine quantization scale or zero point");
    } else if (attr.qnt_type != RKNN_TENSOR_QNT_NONE) {
        return Status::failure(ErrorCode::Unsupported, prefix + "floating-point tensor must not carry integer quantization");
    }
    return Status::success();
}

/* 当前分割架构限定 32 通道 prototype，空间尺寸为模型输入的四分之一。 */
static bool prototypeShape(const Yolo_t& yolo, const rknn_tensor_attr& attr) noexcept
{
    return attr.n_dims == 4U && attr.fmt == RKNN_TENSOR_NCHW
        && attr.dims[0] == 1U && attr.dims[1] == YOLO_MASK_CHANNELS
        && attr.dims[2] == static_cast<std::uint32_t>(yolo.inH / 4)
        && attr.dims[3] == static_cast<std::uint32_t>(yolo.inW / 4);
}

/**
 * @brief 枚举最多 4! 种角色分配，联合名称与通道数确定唯一映射。
 * @note  类别数为 1、4、32、64 时，分数可能与 gate、框或 mask 系数同形。
 *        不能只按通道数先到先得；有唯一解时允许重排，多解时只接受调用处
 *        已核验的完整输出顺序，否则拒绝，避免把分数当坐标或系数读取。
 */
static bool assignRoles(const Yolo_t& yolo, const int* indices, int count,
                        const int* channels, const TensorRole* roles,
                        bool allowDocumentedOrder, int* result) noexcept
{
    int permutation[4] = {0, 1, 2, 3};
    int solutions = 0;
    do {
        bool match = true;
        for (int role = 0; role < count; ++role) {
            const auto& attr = yolo.outputAttrs[indices[permutation[role]]];
            const TensorRole named = tensorRole(attr);
            if (attr.dims[1] != static_cast<std::uint32_t>(channels[role])
                || (named != TensorRole::Unknown && named != roles[role])) {
                match = false;
                break;
            }
        }
        if (match) {
            ++solutions;
            for (int role = 0; role < count; ++role)
                result[role] = indices[permutation[role]];
        }
    } while (std::next_permutation(permutation, permutation + count));
    if (solutions == 1)
        return true;
    if (solutions == 0 || !allowDocumentedOrder)
        return false;
    for (int role = 0; role < count; ++role) {
        const auto& attr = yolo.outputAttrs[indices[role]];
        const TensorRole named = tensorRole(attr);
        if (attr.dims[1] != static_cast<std::uint32_t>(channels[role])
            || (named != TensorRole::Unknown && named != roles[role]))
            return false;
        result[role] = indices[role];
    }
    std::printf("[YOLO] Equal-shaped output roles use the documented export order.\n");
    return true;
}

/**
 * @brief 识别支持的分割输出契约，并缓存各角色索引与预测数量。
 * @note  先找唯一 prototype；两输出再区分 packed/E2E，四输出匹配 raw，
 *        十/十三输出按 stride 8、16、32 分组匹配 DFL，后者可带 gate。
 *        该函数不依据类别语义映射 ID，也不把不支持的输出猜成相近布局。
 */
static rk_yolo::Status matchOutputLayout(Yolo_t& yolo)
{
    using rk_yolo::ErrorCode;
    using rk_yolo::Status;
    const auto unsupported = [&yolo]() {
        return Status::failure(ErrorCode::Unsupported,
            "Unsupported or ambiguous output layout for numClasses=" + std::to_string(yolo.numClasses)
            + ". Expected 2 packed/E2E, 4 raw-split, or 10/13 split-DFL outputs; "
              "equal-shaped roles need explicit names or the documented output order.");
    };
    for (int i = 0; i < yolo.nOut; ++i) {
        const auto& attr = yolo.outputAttrs[i];
        if (prototypeShape(yolo, attr)) {
            const auto role = tensorRole(attr);
            if (yolo.idxProto >= 0 || (role != TensorRole::Unknown && role != TensorRole::Proto))
                return unsupported();
            yolo.idxProto = i;
            yolo.maskH = static_cast<int>(attr.dims[2]);
            yolo.maskW = static_cast<int>(attr.dims[3]);
        } else if (tensorRole(attr) == TensorRole::Proto) {
            return Status::failure(ErrorCode::Unsupported, "Prototype must be NCHW [1,32,inputH/4,inputW/4]");
        }
    }
    if (yolo.idxProto < 0)
        return Status::failure(ErrorCode::Unsupported, "Missing NCHW [1,32,inputH/4,inputW/4] mask prototype");

    if (yolo.nOut == 2) {
        const int index = 1 - yolo.idxProto;
        const auto& attr = yolo.outputAttrs[index];
        if (attr.n_dims != 3U || attr.dims[0] != 1U)
            return unsupported();
        const bool packed = attr.dims[1] == static_cast<std::uint32_t>(4 + yolo.numClasses + YOLO_MASK_CHANNELS);
        const bool endToEnd = attr.dims[2] == static_cast<std::uint32_t>(6 + YOLO_MASK_CHANNELS);
        if (packed == endToEnd)
            return unsupported(); // 同时匹配时无法确定通道轴与条目轴，必须拒绝。
        const auto role = tensorRole(attr);
        if (role != TensorRole::Unknown)
            return unsupported();
        yolo.idxDetection = index;
        yolo.outputLayout = packed ? YOLO_OUTPUT_PACKED_SEG : YOLO_OUTPUT_END2END_SEG;
        yolo.nPredictions = static_cast<int>(attr.dims[packed ? 2 : 1]);
        if (endToEnd && (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8)
            && attr.scale > 0.05F)
            return Status::failure(ErrorCode::Unsupported,
                "E2E integer output scale is too coarse for 0..1 scores; export FP16/FP32 or raw outputs");
        return Status::success();
    }

    /* raw 形状按 [1,通道数,N] 解释，三类输出必须指向同一批 N 个预测。
       角色同形且无名称时，只有 boxes、scores、coeff、proto 完整顺序可用。 */
    if (yolo.nOut == 4) {
        int indices[3]{};
        int count = 0;
        int predictions = -1;
        for (int i = 0; i < yolo.nOut; ++i) {
            if (i == yolo.idxProto) continue;
            const auto& attr = yolo.outputAttrs[i];
            if (attr.n_dims != 3U || attr.dims[0] != 1U)
                return unsupported();
            if (predictions < 0) predictions = static_cast<int>(attr.dims[2]);
            if (attr.dims[2] != static_cast<std::uint32_t>(predictions))
                return Status::failure(ErrorCode::Unsupported, "Raw boxes, scores and coefficients must share the same N");
            indices[count++] = i;
        }
        const int channels[3] = {4, yolo.numClasses, YOLO_MASK_CHANNELS};
        const TensorRole roles[3] = {TensorRole::Box, TensorRole::Scores, TensorRole::Coeff};
        int matched[3]{};
        if (!assignRoles(yolo, indices, 3, channels, roles, yolo.idxProto == 3, matched))
            return unsupported();
        yolo.idxRawBox = matched[0];
        yolo.idxRawClass = matched[1];
        yolo.idxRawMask = matched[2];
        yolo.nPredictions = predictions;
        yolo.outputLayout = YOLO_OUTPUT_RAW_SEG;
        return Status::success();
    }

    if (yolo.nOut != 10 && yolo.nOut != 13)
        return unsupported();
    /* DFL 每尺度依次为 box、scores、可选 gate、coeff，最后为 prototype；
       先按真实网格尺寸分组，再验证角色，避免类别数碰撞破坏尺度归属。 */
    const bool withGate = yolo.nOut == 13;
    const int perScale = withGate ? 4 : 3;
    int grouped[3][4]{};
    int counts[3]{};
    for (int i = 0; i < yolo.nOut; ++i) {
        if (i == yolo.idxProto) continue;
        const auto& attr = yolo.outputAttrs[i];
        if (attr.n_dims != 4U || attr.fmt != RKNN_TENSOR_NCHW || attr.dims[0] != 1U)
            return unsupported();
        const int h = static_cast<int>(attr.dims[2]), w = static_cast<int>(attr.dims[3]);
        if (yolo.inW % w != 0 || yolo.inH % h != 0 || yolo.inW / w != yolo.inH / h)
            return unsupported();
        const int stride = yolo.inW / w;
        const int scale = stride == 8 ? 0 : stride == 16 ? 1 : stride == 32 ? 2 : -1;
        if (scale < 0 || counts[scale] >= perScale)
            return unsupported();
        yolo.gridH[scale] = h;
        yolo.gridW[scale] = w;
        grouped[scale][counts[scale]++] = i;
    }
    for (int scale = 0; scale < 3; ++scale) {
        if (counts[scale] != perScale)
            return unsupported();
        const int channels[4] = {64, yolo.numClasses, withGate ? 1 : YOLO_MASK_CHANNELS, YOLO_MASK_CHANNELS};
        const TensorRole roles[4] = {TensorRole::Box, TensorRole::Scores,
                                     withGate ? TensorRole::Gate : TensorRole::Coeff, TensorRole::Coeff};
        bool documented = yolo.idxProto == yolo.nOut - 1;
        for (int j = 0; j < perScale; ++j)
            documented = documented && grouped[scale][j] == scale * perScale + j;
        int matched[4]{};
        if (!assignRoles(yolo, grouped[scale], perScale, channels, roles, documented, matched))
            return unsupported();
        yolo.idxBox[scale] = matched[0];
        yolo.idxClass[scale] = matched[1];
        yolo.idxGate[scale] = withGate ? matched[2] : -1;
        yolo.idxMask[scale] = matched[withGate ? 3 : 2];
    }
    yolo.nScale = 3;
    yolo.outputLayout = YOLO_OUTPUT_SPLIT_DFL;
    return Status::success();
}

/**
 * @brief 建立 RKNN 上下文、验证模型契约并分配复用缓冲区。
 * @note  numClasses 由调用方显式提供，必须与导出模型一致；不从类别名猜测。
 *        本函数中途失败由公共 load() 统一释放已取得的资源，并返回具体原因。
 */
static rk_yolo::Status Yolo_Load(Yolo_t* yolo, const char* modelPath, int numClasses)
{
    using rk_yolo::ErrorCode;
    using rk_yolo::Status;
    yolo->numClasses = numClasses;
    if (rknn_init(&yolo->ctx, const_cast<char*>(modelPath), 0, 0, nullptr) != RKNN_SUCC)
        return Status::failure(ErrorCode::Hardware, "RKNN failed to initialize the model");

    const char* coreMaskName = nullptr;
    const rknn_core_mask coreMask = requestedCoreMask(&coreMaskName);
    if (rknn_set_core_mask(yolo->ctx, coreMask) != RKNN_SUCC)
        std::printf("[YOLO] Requested core mask %s was rejected; using runtime scheduling.\n", coreMaskName);
    rknn_sdk_version sdkVersion{};
    if (rknn_query(yolo->ctx, RKNN_QUERY_SDK_VERSION, &sdkVersion, sizeof(sdkVersion)) == RKNN_SUCC)
        std::printf("[YOLO] RKNN API=%.*s, driver=%.*s\n",
            static_cast<int>(sizeof(sdkVersion.api_version)), sdkVersion.api_version,
            static_cast<int>(sizeof(sdkVersion.drv_version)), sdkVersion.drv_version);

    rknn_input_output_num ioNum{};
    if (rknn_query(yolo->ctx, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)) != RKNN_SUCC)
        return Status::failure(ErrorCode::Hardware, "RKNN failed to query input/output counts");
    if (ioNum.n_input != 1U || ioNum.n_output == 0U || ioNum.n_output > rk_yolo::YOLO_MAX_OUTPUTS)
        return Status::failure(ErrorCode::Unsupported, "Model requires one input and at most 16 outputs");
    yolo->nOut = static_cast<int>(ioNum.n_output);

    rknn_tensor_attr input{};
    if (rknn_query(yolo->ctx, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)) != RKNN_SUCC)
        return Status::failure(ErrorCode::Hardware, "RKNN failed to query input attributes");
    const auto inputStatus = validateTensor(input, true);
    if (!inputStatus) return inputStatus;
    /* 模型逻辑输入可为 NHWC 或 NCHW；这里仅读取正确的 H/W，
       实际送入 RKNN 的缓冲区始终是 NHWC、UINT8 RGB，由运行时转换。 */
    const bool nhwc = input.fmt == RKNN_TENSOR_NHWC && input.n_dims == 4U
        && input.dims[0] == 1U && input.dims[3] == 3U;
    const bool nchw = input.fmt == RKNN_TENSOR_NCHW && input.n_dims == 4U
        && input.dims[0] == 1U && input.dims[1] == 3U;
    if (!nhwc && !nchw)
        return Status::failure(ErrorCode::Unsupported,
            "Input must be NHWC [1,H,W,3] or NCHW [1,3,H,W], with batch one");
    yolo->inH = static_cast<int>(input.dims[nhwc ? 1 : 2]);
    yolo->inW = static_cast<int>(input.dims[nhwc ? 2 : 3]);
    if (yolo->inH % 4 != 0 || yolo->inW % 4 != 0)
        return Status::failure(ErrorCode::Unsupported, "Input H/W must be positive and divisible by four");
    for (int i = 0; i < yolo->nOut; ++i) {
        auto& attr = yolo->outputAttrs[i];
        attr.index = static_cast<std::uint32_t>(i);
        if (rknn_query(yolo->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC)
            return Status::failure(ErrorCode::Hardware, "RKNN failed to query output " + std::to_string(i));
        const auto status = validateTensor(attr, false);
        if (!status) return status;
    }
    const auto layout = matchOutputLayout(*yolo);
    if (!layout) return layout;

    /* 仅当所有输出同为 INT8 或 FP16 才使用对应原生读取路径；
       混合类型、UINT8 或强制回退选项均让 RKNN 统一返回 FP32。 */
    yolo->useNativeInt8 = 1;
    yolo->useNativeFp16 = 1;
    for (int i = 0; i < yolo->nOut; ++i) {
        const auto& attr = yolo->outputAttrs[i];
        if (attr.type != RKNN_TENSOR_INT8) yolo->useNativeInt8 = 0;
        if (attr.type != RKNN_TENSOR_FLOAT16) yolo->useNativeFp16 = 0;
    }
    const char* forceFloat = std::getenv("RK_YOLO_RKNN_FLOAT_OUTPUT");
    if (forceFloat != nullptr && std::strcmp(forceFloat, "1") == 0)
        yolo->useNativeInt8 = yolo->useNativeFp16 = 0;
    std::printf("[YOLO] classes=%d input=%dx%d layout=%d outputs=%s\n", yolo->numClasses,
        yolo->inW, yolo->inH, static_cast<int>(yolo->outputLayout),
        yolo->useNativeInt8 ? "native INT8" : yolo->useNativeFp16 ? "native FP16" : "FLOAT32");

    /* 加载期按模型尺寸准备 RGB、mask 行累加、底点统计与候选容器。
       这些容量覆盖单帧所需上限，逐帧推理复用，不随检测数量反复扩容。 */
    const std::size_t rgbBytes = static_cast<std::size_t>(yolo->inW) * yolo->inH * 3U;
    const std::size_t maskPixels = static_cast<std::size_t>(yolo->maskW) * yolo->maskH;
    if (maskPixels > static_cast<std::size_t>(INT_MAX))
        return Status::failure(ErrorCode::Unsupported, "Mask prototype exceeds supported pixel count");
    yolo->letterboxStorage.resize(rgbBytes, 114U);
    yolo->maskPoints.reserve(maskPixels);
    yolo->maskRowLogits.resize(static_cast<std::size_t>(yolo->maskW));
    yolo->maskRowCounts.resize(static_cast<std::size_t>(yolo->maskH));
    std::size_t candidateCapacity = 0U;
    if (yolo->outputLayout != YOLO_OUTPUT_SPLIT_DFL)
        candidateCapacity = static_cast<std::size_t>(yolo->nPredictions);
    else
        for (int scale = 0; scale < yolo->nScale; ++scale)
            candidateCapacity += static_cast<std::size_t>(yolo->gridH[scale]) * yolo->gridW[scale];
    auto workspace = std::make_unique<YoloWorkspace>();
    workspace->candidateCapacity = candidateCapacity;
    workspace->outputs.resize(static_cast<std::size_t>(yolo->nOut));
    workspace->candidates.resize(candidateCapacity);
    yolo->workspace = workspace.release();
    return Status::success();
}

/**
 * @brief 销毁上下文、释放工作区并恢复默认状态；允许重复调用。
 * @note  释放同时清空模型尺寸、布局与自定义选项，新模型加载后应重新设置
 *        所需的 NMS、色彩标准、ROI 与统计开关。
 */
static void Yolo_Release(Yolo_t* yolo)
{
    if (yolo == nullptr) return;
    if (yolo->ctx != 0) rknn_destroy(yolo->ctx);
    delete static_cast<YoloWorkspace*>(yolo->workspace);
    *yolo = Yolo_t{};
}


/**
 * @brief 同步处理一帧 NV12，返回结果数量；失败返回负数并记录阶段错误。
 * @note  外层 detect() 先验证输入字节长度，这里继续检查阈值、偶数尺寸与
 *        stride。候选和掩膜仍借用本次 RKNN 输出，必须完成后处理后再释放。
 */
static int Yolo_Detect(Yolo_t* yolo, const uint8_t* nv12, int width, int height,
                       int strideW, int strideH, float confThr, float gateThr,
                       SegRegion_t* out, int maxOut)
{
    if (yolo != nullptr) {
        yolo->lastOutputTruncated = 0;
        yolo->lastError = YOLO_ERROR_NONE;
        yolo->lastTiming = YoloTiming_t{};
    }
    if (yolo == nullptr || yolo->ctx == 0 || yolo->workspace == nullptr
        || nv12 == nullptr || width <= 0 || height <= 0 || out == nullptr
        || maxOut <= 0 || maxOut > YOLO_MAX_REGIONS
        || !isfinite(confThr) || !isfinite(gateThr)
        || confThr < 0.0F || confThr > 1.0F || gateThr < 0.0F)
        return detectError(yolo, YOLO_ERROR_INVALID_ARGUMENT);
    const int sW = strideW > 0 ? strideW : width, sH = strideH > 0 ? strideH : height;
    if (sW < width || sH < height || (width & 1) || (height & 1) || (sW & 1) || (sH & 1))
        return detectError(yolo, YOLO_ERROR_INVALID_STRIDE);
    auto* ws = static_cast<YoloWorkspace*>(yolo->workspace);
    /* 阶段一：保留本帧 letterbox 几何，供框和掩膜共同做逆变换。 */
    float scale = 0.0F, padX = 0.0F, padY = 0.0F;
    const double preprocessStart = yolo->profilingEnabled ? profileMilliseconds() : 0.0;
    if (!letterbox(yolo, nv12, width, height, sW, sH, &scale, &padX, &padY))
        return detectError(yolo, YOLO_ERROR_PREPROCESS);
    if (yolo->profilingEnabled)
        yolo->lastTiming.preprocessMilliseconds = profileMilliseconds() - preprocessStart;
    /* 阶段二：提交模型尺寸的 RGB 数据并执行 RKNN。
       推理计时包含输出获取和缓冲区检查，不能只当作 NPU 算子耗时。 */
    const double inferenceStart = yolo->profilingEnabled ? profileMilliseconds() : 0.0;
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<std::uint32_t>(yolo->letterboxStorage.size());
    input.buf = yolo->letterboxStorage.data();
    if (rknn_inputs_set(yolo->ctx, 1, &input) != RKNN_SUCC)
        return detectError(yolo, YOLO_ERROR_SET_INPUT);
    if (rknn_run(yolo->ctx, nullptr) != RKNN_SUCC)
        return detectError(yolo, YOLO_ERROR_RUN);

    auto* outs = ws->outputs.data();
    std::memset(outs, 0, static_cast<std::size_t>(yolo->nOut) * sizeof(rknn_output));
    const bool floatOutput = !yolo->useNativeInt8 && !yolo->useNativeFp16;
    for (int i = 0; i < yolo->nOut; ++i) {
        outs[i].index = static_cast<std::uint32_t>(i);
        outs[i].want_float = floatOutput;
    }
    if (rknn_outputs_get(yolo->ctx, yolo->nOut, outs, nullptr) != RKNN_SUCC)
        return detectError(yolo, YOLO_ERROR_GET_OUTPUT);
    /* 只有成功获取输出后才建立释放守卫；后续尺寸检查失败或提前返回时，
       仍归还本次输出。返回的 DetectionList 不持有任何 RKNN 缓冲区指针。 */
    struct OutputsGuard {
        Yolo_t* yolo;
        rknn_output* outs;
        ~OutputsGuard() { rknn_outputs_release(yolo->ctx, yolo->nOut, outs); }
    } guard{yolo, outs};
    for (int i = 0; i < yolo->nOut; ++i) {
        const std::size_t elementBytes = floatOutput ? sizeof(float)
            : tensorElementBytes(yolo->outputAttrs[i].type);
        const std::size_t required = static_cast<std::size_t>(yolo->outputAttrs[i].n_elems) * elementBytes;
        if (outs[i].buf == nullptr || outs[i].size < required)
            return detectError(yolo, YOLO_ERROR_OUTPUT_DATA);
    }
    if (yolo->profilingEnabled)
        yolo->lastTiming.inferenceMilliseconds = profileMilliseconds() - inferenceStart;
    /* 阶段三：按已确认的布局解码，所有候选统一为模型像素 xyxy。
       DFL/raw/packed 选择每个网格或条目的最高分类分数；E2E 沿用条目中的
       类别与分数。每处最多生成一个候选，不做类别名映射。 */
    const double postStart = yolo->profilingEnabled ? profileMilliseconds() : 0.0;
    Cand* cands = ws->candidates.data();
    std::size_t nCands = 0;
    const auto append = [&](float x1, float y1, float x2, float y2, float score,
                            int cls, int scaleIndex, int offset) {
        if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2)
            || x2 <= x1 || y2 <= y1 || nCands >= ws->candidateCapacity)
            return;
        cands[nCands++] = Cand{x1, y1, x2, y2, score, cls, scaleIndex, offset};
    };

    /* E2E 每条记录为 xyxy、score、class ID、32 个 mask 系数。
       转整数前先检查 class ID 范围，避免异常浮点值进入整数转换。 */
    if (yolo->outputLayout == YOLO_OUTPUT_END2END_SEG) {
        constexpr int fields = 6 + YOLO_MASK_CHANNELS;
        for (int prediction = 0; prediction < yolo->nPredictions; ++prediction) {
            const std::size_t base = static_cast<std::size_t>(prediction) * fields;
            const float score = outputValue(yolo, outs, yolo->idxDetection, base + 4U);
            const float value = outputValue(yolo, outs, yolo->idxDetection, base + 5U);
            if (!isfinite(score) || score <= confThr || score > 1.001F
                || !isfinite(value) || value < -0.25F || value > yolo->numClasses - 1 + 0.25F)
                continue;
            const int cls = static_cast<int>(lroundf(value));
            if (cls < 0 || cls >= yolo->numClasses || fabsf(value - static_cast<float>(cls)) > 0.25F)
                continue;
            append(outputValue(yolo, outs, yolo->idxDetection, base),
                   outputValue(yolo, outs, yolo->idxDetection, base + 1U),
                   outputValue(yolo, outs, yolo->idxDetection, base + 2U),
                   outputValue(yolo, outs, yolo->idxDetection, base + 3U),
                   score, cls, -1, prediction);
        }
    } else if (yolo->outputLayout == YOLO_OUTPUT_RAW_SEG
               || yolo->outputLayout == YOLO_OUTPUT_PACKED_SEG) {
        /* raw 将框/分数分开；packed 的分数从第 4 通道开始。
           两者框均为模型像素 cxcywh，分数已激活，不再乘 objectness。 */
        const bool packed = yolo->outputLayout == YOLO_OUTPUT_PACKED_SEG;
        const int boxIndex = packed ? yolo->idxDetection : yolo->idxRawBox;
        const int classIndex = packed ? yolo->idxDetection : yolo->idxRawClass;
        const std::size_t plane = static_cast<std::size_t>(yolo->nPredictions);
        for (int prediction = 0; prediction < yolo->nPredictions; ++prediction) {
            const std::size_t offset = static_cast<std::size_t>(prediction);
            int bestClass = -1;
            float bestScore = -1.0F;
            for (int cls = 0; cls < yolo->numClasses; ++cls) {
                const std::size_t channel = static_cast<std::size_t>(cls + (packed ? 4 : 0));
                const float score = outputValue(yolo, outs, classIndex, channel * plane + offset);
                if (isfinite(score) && score >= 0.0F && score <= 1.001F && score > bestScore) {
                    bestClass = cls;
                    bestScore = score;
                }
            }
            if (bestClass < 0 || bestScore <= confThr) continue;
            const float cx = outputValue(yolo, outs, boxIndex, offset);
            const float cy = outputValue(yolo, outs, boxIndex, plane + offset);
            const float w = outputValue(yolo, outs, boxIndex, 2U * plane + offset);
            const float h = outputValue(yolo, outs, boxIndex, 3U * plane + offset);
            if (!isfinite(cx) || !isfinite(cy) || !isfinite(w) || !isfinite(h) || w <= 0.0F || h <= 0.0F)
                continue;
            append(cx - w * 0.5F, cy - h * 0.5F, cx + w * 0.5F, cy + h * 0.5F,
                   bestScore, bestClass, -1, prediction);
        }
    } else {
        /* DFL 网格的四边距离乘 stride 后转为模型像素。
           gate 只用于跳过低响应网格；没有 gate 的十输出模型跳过此筛选。 */
        for (int s = 0; s < yolo->nScale; ++s) {
            const int gw = yolo->gridW[s], gh = yolo->gridH[s], gridLen = gw * gh;
            const float stride = static_cast<float>(yolo->inW) / gw;
            for (int y = 0; y < gh; ++y) {
                for (int x = 0; x < gw; ++x) {
                    const int offset = y * gw + x;
                    if (yolo->idxGate[s] >= 0) {
                        const float gate = outputValue(yolo, outs, yolo->idxGate[s], static_cast<std::size_t>(offset));
                        if (!isfinite(gate) || gate < gateThr) continue;
                    }
                    int cls = -1;
                    float bestScore = -1.0F;
                    for (int c = 0; c < yolo->numClasses; ++c) {
                        const float score = outputValue(yolo, outs, yolo->idxClass[s],
                            static_cast<std::size_t>(c) * gridLen + offset);
                        if (isfinite(score) && score >= 0.0F && score <= 1.001F && score > bestScore) {
                            cls = c;
                            bestScore = score;
                        }
                    }
                    if (cls < 0 || bestScore <= confThr) continue;
                    float logits[64]{};
                    bool finite = true;
                    for (int k = 0; k < 64; ++k) {
                        logits[k] = outputValue(yolo, outs, yolo->idxBox[s],
                            static_cast<std::size_t>(k) * gridLen + offset);
                        finite = finite && isfinite(logits[k]);
                    }
                    if (!finite) continue;
                    float box[4]{};
                    computeDfl(logits, 16, box);
                    append((x + 0.5F - box[0]) * stride, (y + 0.5F - box[1]) * stride,
                           (x + 0.5F + box[2]) * stride, (y + 0.5F + box[3]) * stride,
                           bestScore, cls, s, offset);
                }
            }
        }
    }

    /* 阶段四：先按分数排序，再做同类贪心 NMS；同分时用来源位置确定顺序。 */
    std::sort(cands, cands + nCands, [](const Cand& a, const Cand& b) noexcept {
        if (a.score != b.score) return a.score > b.score;
        if (a.scaleIndex != b.scaleIndex) return a.scaleIndex < b.scaleIndex;
        return a.gridOffset < b.gridOffset;
    });
    // 只与最多 64 个已接受框比较，同类重叠才抑制，不同类保留各自 ID。
    // 该暂存数组随结果容量固定，不按类别数建立大数组。
    const Cand* accepted[YOLO_MAX_REGIONS]{};
    int count = 0;
    const float invScale = 1.0F / scale;
    for (std::size_t i = 0; i < nCands; ++i) {
        const Cand& c = cands[i];
        bool suppressed = false;
        for (int previous = 0; previous < count; ++previous) {
            const Cand& a = *accepted[previous];
            if (a.cls == c.cls && boxIou(a.x1, a.y1, a.x2, a.y2, c.x1, c.y1, c.x2, c.y2) > yolo->nmsIouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        /* 先减模型像素补边，再除 scale 回到原图，最后除原图宽高归一化。
           超出有效图像的部分裁掉，完全落在补边里的退化框不作为结果。 */
        const float x1 = fmaxf(0.0F, (c.x1 - padX) * invScale / width);
        const float y1 = fmaxf(0.0F, (c.y1 - padY) * invScale / height);
        const float x2 = fminf(1.0F, (c.x2 - padX) * invScale / width);
        const float y2 = fminf(1.0F, (c.y2 - padY) * invScale / height);
        if (x2 <= x1 || y2 <= y1) continue;
        /* 容量已满且确有下一个有效、未抑制的框时才标记截断；
           恰好 64 个结果不自动等于截断。调用方需据此判断结果是否完整。 */
        if (count >= maxOut) {
            yolo->lastOutputTruncated = 1;
            break;
        }
        SegRegion_t& r = out[count];
        r = SegRegion_t{};
        r.classId = c.cls;
        r.score = c.score;
        r.x1 = x1; r.y1 = y1; r.x2 = x2; r.y2 = y2;
        r.cx = (x1 + x2) * 0.5F;
        r.cy = (y1 + y2) * 0.5F;
        r.bottomX = r.cx;
        r.bottomY = y2;
        r.area = (x2 - x1) * (y2 - y1);
        /* 阶段五：在模型输出仍有效时计算所有保留类别的掩膜统计。
           mask 耗时已包含在 post 中，两者不可重复相加。 */
        const double maskStart = yolo->profilingEnabled ? profileMilliseconds() : 0.0;
        r.maskValid = computeMaskCentroid(yolo, &c, outs, scale, padX, padY, width, height,
            r.cx, r.cy, 0.0F, r.bottomX, r.bottomY,
            &r.cx, &r.cy, &r.maskArea, &r.bottomX, &r.bottomY, &r.roiMaskOverlap) != 0;
        if (yolo->profilingEnabled) {
            yolo->lastTiming.maskMilliseconds += profileMilliseconds() - maskStart;
            ++yolo->lastTiming.maskCandidates;
        }
        accepted[count++] = &c;
    }
    if (yolo->profilingEnabled)
        yolo->lastTiming.postprocessMilliseconds = profileMilliseconds() - postStart;
    return count;
}

} // namespace

namespace rk_yolo {

/* PImpl 使公共头不暴露 RKNN/RGA 句柄，检测结果仍是可独立复制的值类型。 */
struct YoloDetector::Impl { Yolo_t storage{}; };

YoloDetector::YoloDetector() noexcept : impl_(new (std::nothrow) Impl{}) {}
YoloDetector::~YoloDetector() noexcept { release(); }
YoloDetector::YoloDetector(YoloDetector&& other) noexcept = default;
YoloDetector& YoloDetector::operator=(YoloDetector&& other) noexcept {
    if (this != &other) { release(); impl_ = std::move(other.impl_); }
    return *this;
}

/**
 * @brief 校验调用参数并加载模型，将内部错误转换为 Status。
 * @note  参数有效后会先释放旧模型；新模型建立中途失败则清理已取得的资源。
 *        内部分配和错误文本可能失败，异常在此转为错误码，不越过 noexcept。
 */
Status YoloDetector::load(std::string_view modelPath, int numClasses) noexcept
{
    try {
        if (modelPath.empty() || modelPath.find('\0') != std::string_view::npos
            || numClasses < 1 || numClasses > YOLO_MAX_CLASSES)
            return Status::failure(ErrorCode::InvalidArgument,
                "Provide a nonempty model path and numClasses in 1..4096");
        if (!impl_) impl_.reset(new (std::nothrow) Impl{});
        if (!impl_) return Status::failure(ErrorCode::OutOfMemory);
        release();
        const std::string path(modelPath);
        auto status = Yolo_Load(&impl_->storage, path.c_str(), numClasses);
        if (!status) release();
        return status;
    } catch (const std::bad_alloc&) {
        release();
        return Status::failure(ErrorCode::OutOfMemory);
    } catch (...) {
        release();
        return Status::failure(ErrorCode::Internal);
    }
}
void YoloDetector::release() noexcept { if (impl_) Yolo_Release(&impl_->storage); }
bool YoloDetector::isLoaded() const noexcept { return impl_ && Yolo_IsLoaded(&impl_->storage) != 0; }
int YoloDetector::classCount() const noexcept { return isLoaded() ? impl_->storage.numClasses : 0; }
int YoloDetector::inputWidth() const noexcept { return isLoaded() ? impl_->storage.inW : 0; }
int YoloDetector::inputHeight() const noexcept { return isLoaded() ? impl_->storage.inH : 0; }
YoloOutputLayout YoloDetector::outputLayout() const noexcept {
    return isLoaded() ? impl_->storage.outputLayout : YoloOutputLayout::Unknown;
}
Status YoloDetector::setMaskRoi(float x1, float y1, float x2, float y2) noexcept {
    return impl_ && Yolo_SetMaskRoi(&impl_->storage, x1, y1, x2, y2) != 0
        ? Status::success() : Status::failure(ErrorCode::InvalidArgument);
}
Status YoloDetector::setNmsIou(float threshold) noexcept {
    return impl_ && Yolo_SetNmsIou(&impl_->storage, threshold) != 0
        ? Status::success() : Status::failure(ErrorCode::InvalidArgument);
}
Status YoloDetector::setColorSpace(CameraColorSpace colorSpace) noexcept {
    return impl_ && Yolo_SetColorSpace(&impl_->storage, colorSpace) != 0
        ? Status::success() : Status::failure(ErrorCode::InvalidArgument);
}
Status YoloDetector::setBottomRefinement(bool enabled) noexcept {
    return impl_ && Yolo_SetBottomRefinement(&impl_->storage, enabled ? 1 : 0) != 0
        ? Status::success() : Status::failure(ErrorCode::InvalidArgument);
}
/**
 * @brief 验证完整 NV12 帧并返回本次推理的独立结果。
 * @note  FrameView 不拥有图像内存，调用方必须保证同步调用结束前缓冲区有效。
 *        字节校验包含 stride 的 Y/UV 平面及溢出检查；失败不调用 RGA。
 *        每次调用重置本帧错误、耗时与截断标记，空结果与调用失败由 Result 区分。
 */
Result<DetectionList> YoloDetector::detect(FrameView frame, float confidenceThreshold,
                                          float gateThreshold) noexcept
{
    try {
        if (impl_) {
            impl_->storage.lastError = YoloError::None;
            impl_->storage.lastOutputTruncated = 0;
            impl_->storage.lastTiming = YoloTiming{};
        }
        if (!isLoaded())
            return Result<DetectionList>::failure(ErrorCode::NotInitialized, "Load a model before detect()");
        if (!isValidNv12Frame(frame)) {
            impl_->storage.lastError = YoloError::InvalidArgument;
            return Result<DetectionList>::failure(ErrorCode::InvalidArgument, "Invalid or undersized NV12 frame");
        }
        DetectionList detections{};
        std::array<SegRegion_t, YOLO_MAX_REGIONS> output{};
        const int count = Yolo_Detect(&impl_->storage, frame.bytes.data(), frame.geometry.width,
            frame.geometry.height, frame.geometry.strideWidth, frame.geometry.strideHeight,
            confidenceThreshold, gateThreshold, output.data(), static_cast<int>(output.size()));
        if (count < 0)
            return Result<DetectionList>::failure(ErrorCode::Hardware, "YOLO inference failed; inspect lastError()");
        for (int i = 0; i < count; ++i)
            detections.push_back(output[static_cast<std::size_t>(i)]);
        return Result<DetectionList>::success(std::move(detections));
    } catch (const std::bad_alloc&) {
        return Result<DetectionList>::failure(ErrorCode::OutOfMemory);
    } catch (...) {
        return Result<DetectionList>::failure(ErrorCode::Internal);
    }
}
YoloError YoloDetector::lastError() const noexcept { return impl_ ? impl_->storage.lastError : YoloError::InvalidArgument; }
YoloTiming YoloDetector::timing() const noexcept { return impl_ ? impl_->storage.lastTiming : YoloTiming{}; }
bool YoloDetector::outputTruncated() const noexcept { return impl_ && impl_->storage.lastOutputTruncated != 0; }
void YoloDetector::setProfiling(bool enabled) noexcept { if (impl_) impl_->storage.profilingEnabled = enabled ? 1 : 0; }
std::string_view YoloDetector::errorName(YoloError error) noexcept {
    switch (error) {
    case YoloError::None: return "none";
    case YoloError::InvalidArgument: return "invalid_argument";
    case YoloError::InvalidStride: return "invalid_stride";
    case YoloError::Preprocess: return "preprocess";
    case YoloError::SetInput: return "set_input";
    case YoloError::Run: return "run";
    case YoloError::GetOutput: return "get_output";
    case YoloError::OutputData: return "output_data";
    }
    return "unknown";
}

} // namespace rk_yolo
