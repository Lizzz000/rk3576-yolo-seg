// SPDX-License-Identifier: Apache-2.0
/**
 * @file    shared_options.hpp
 * @brief   图片/摄像头示例共享的参数、UTF-8 标签与结果展示接口。
 *
 * 该层不定义模型类别，不加载 RKNN，也不打开摄像头；负责把命令行与标签文件
 * 转为明确的类数、阈值和展示名称，供两个入口复用。
 */
#ifndef RK_YOLO_EXAMPLES_SHARED_OPTIONS_HPP
#define RK_YOLO_EXAMPLES_SHARED_OPTIONS_HPP

#include "yolo/yolo.hpp"
#include <opencv2/core/mat.hpp>
#include <cstdint>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace rk_yolo::example {
/* values 保存带值选项，flags 保存无值开关；get 在键不存在时返回 fallback。 */
struct Arguments final {
    std::map<std::string, std::string> values;
    std::set<std::string> flags;
    bool has(const std::string& key) const;
    std::string get(const std::string& key, const std::string& fallback = {}) const;
};
/* numClasses 必须来自标签行数或显式参数，范围 1..4096。
 * 阈值无单位：confidence/gate 为 [0,1]，nms 为 (0,1]。
 * gate 用于带门控输出的布局；是否使用由模型后处理布局决定。 */
struct CommonOptions final {
    std::string model;
    std::vector<std::string> labels;
    int numClasses{0};
    float confidence{0.25F}, nms{0.45F}, gate{0.0F};
};
/* 仅单独的 --help/-h 视为帮助请求；混入其他参数时继续按普通命令行校验。 */
bool helpRequested(int argc, char** argv);
/**
 * @brief  解析公共选项及入口补充的值选项/开关。
 * @param  output 开始时重置；成功后保存解析结果，不负责验证所有值的业务范围。
 * @return true：选项名和取值结构合法；false：未知、重复、缺值等原因写入 error。
 */
bool parseArguments(int argc, char** argv, const std::set<std::string>& extraValues,
                    const std::set<std::string>& extraFlags,
                    Arguments& output, std::string& error);
/* 只接受十进制数字串，范围为闭区间 [minimum, maximum]；失败不改 output。 */
bool parseUnsigned(const std::string& text, std::uint64_t minimum,
                   std::uint64_t maximum, std::uint64_t& output);
/* 要求完整有限数值，拒绝空白、NaN/Inf 和溢出；最大值包含在内，
 * minimumInclusive 决定是否包含最小值；失败不改 output。 */
bool parseFinite(const std::string& text, double minimum, double maximum,
                 bool minimumInclusive, double& output);
/* --labels 与 --num-classes 至少提供其一；同时提供时标签行数必须严格一致。
 * true 表示公共参数通过，false 通过 error 解释失败；不检查模型是否能加载。 */
bool parseCommon(const Arguments& arguments, CommonOptions& output, std::string& error);
void printCommonUsage(std::ostream& stream);
void printModelInfo(std::ostream& stream, const YoloDetector& detector,
                    const CommonOptions& options);
/* 标签按从 0 开始的 classId 索引；未提供名称或索引越界时返回 class_<id>。 */
std::string classLabel(const CommonOptions& options, int classId);
/* image 为可写原尺寸 BGR 图：绘框、中心圆点和底点十字，不输出完整掩膜。
 * 非 ASCII 名称仅在画面上回退为 class_<id>，控制台保留原 UTF-8 名称。 */
void annotate(cv::Mat& image, const DetectionList& detections, const CommonOptions& options);
/* 输出原图归一化坐标及面积占比；不会把掩膜统计误当作掩膜像素输出。 */
void printDetections(std::ostream& stream, const DetectionList& detections,
                     const CommonOptions& options);
} // namespace rk_yolo::example
#endif
