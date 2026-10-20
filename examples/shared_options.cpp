// SPDX-License-Identifier: Apache-2.0
/**
 * @file    shared_options.cpp
 * @brief   共享参数校验、UTF-8 标签读取、通用配色与检测结果展示实现。
 */
#include "shared_options.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <utility>

namespace rk_yolo::example {
namespace {
/* 使用 Unicode 码点判断空白名，避免中文全角空格等内容被当成有效类别名。 */
bool unicodeSpace(std::uint32_t cp)
{
    return cp == 0x20 || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85
        || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A)
        || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
/* 逐码点校验 UTF-8，不依赖系统 locale：拒绝过长编码、代理区/越界码点、
 * ASCII 控制字符以及空行、纯 Unicode 空白名；名称内容不按内置类别改写。 */
bool validLabel(const std::string& text)
{
    bool visible = false;
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i++]);
        std::uint32_t cp = 0, minimum = 0;
        unsigned int remaining = 0;
        if (lead < 0x80U)
            cp = lead;
        else if ((lead & 0xE0U) == 0xC0U) {
            cp = lead & 0x1FU; remaining = 1; minimum = 0x80;
        } else if ((lead & 0xF0U) == 0xE0U) {
            cp = lead & 0x0FU; remaining = 2; minimum = 0x800;
        } else if ((lead & 0xF8U) == 0xF0U) {
            cp = lead & 0x07U; remaining = 3; minimum = 0x10000;
        } else
            return false;
        if (remaining > text.size() - i)
            return false;
        for (unsigned int j = 0; j < remaining; ++j) {
            const auto next = static_cast<unsigned char>(text[i++]);
            if ((next & 0xC0U) != 0x80U)
                return false;
            cp = (cp << 6U) | (next & 0x3FU);
        }
        if (cp < minimum || cp > 0x10FFFFU || (cp >= 0xD800U && cp <= 0xDFFFU)
            || cp < 0x20 || cp == 0x7F)
            return false;
        visible = visible || !unicodeSpace(cp);
    }
    return visible;
}
/**
 * @brief  按行读取类别名；文件第一行对应类别 0，最多 4096 类。
 * @note   二进制读取后仅去掉文件起始 BOM 与每行末尾 CR，以同时支持 LF/CRLF；
 *         保留标签原文，不跳过空行，否则后续类别编号会错位。
 * @return true：得到非空标签列表；false：文件、编码或行内容错误写入 error。
 */
bool readLabels(const std::string& path, std::vector<std::string>& output, std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "Cannot open UTF-8 labels file: " + path;
        return false;
    }
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (lineNumber == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0)
            line.erase(0, 3);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!validLabel(line)) {
            error = "Invalid UTF-8, empty, whitespace-only or control-character label at line "
                + std::to_string(lineNumber) + ".";
            return false;
        }
        if (output.size() >= 4096U) {
            error = "Labels file has more than 4096 classes.";
            return false;
        }
        output.push_back(line);
    }
    if (input.bad() || (!input.eof() && input.fail())) {
        error = "Cannot finish reading labels file: " + path;
        return false;
    }
    if (output.empty()) {
        error = "Labels file must contain 1..4096 nonempty UTF-8 names.";
        return false;
    }
    return true;
}
/* 将原图归一化位置映射到有效像素索引 [0, extent-1]，绘制时不访问图像外部。
 * extent 来自有效图像宽/高；非有限位置在显示层回退到 0。 */
int pixel(float normalized, int extent)
{
    if (!std::isfinite(normalized))
        return 0;
    return static_cast<int>(std::lround(std::clamp(normalized, 0.0F, 1.0F)
                                      * static_cast<float>(extent - 1)));
}
/* 配色只由类别编号生成，同一编号保持同色，无任何固定类别或颜色语义。
 * 返回 OpenCV 的 BGR 分量，绘制用颜色与模型输入色彩转换相互独立。 */
cv::Scalar classColor(int classId)
{
    std::uint32_t hash = static_cast<std::uint32_t>(classId) + 0x9E3779B9U;
    hash = (hash ^ (hash >> 16U)) * 0x85EBCA6BU;
    hash = (hash ^ (hash >> 13U)) * 0xC2B2AE35U;
    hash ^= hash >> 16U;
    return cv::Scalar(64 + (hash & 0xBFU), 64 + ((hash >> 8U) & 0xBFU),
                      64 + ((hash >> 16U) & 0xBFU));
}
const char* layoutName(YoloOutputLayout layout)
{
    switch (layout) {
    case YoloOutputLayout::SplitDfl: return "split_dfl";
    case YoloOutputLayout::RawSeg: return "raw_seg";
    case YoloOutputLayout::PackedSeg: return "packed_seg";
    case YoloOutputLayout::End2EndSeg: return "end_to_end_seg";
    case YoloOutputLayout::Unknown: return "unknown";
    }
    return "unknown";
}
} // namespace

bool Arguments::has(const std::string& key) const
{
    return values.count(key) != 0 || flags.count(key) != 0;
}
std::string Arguments::get(const std::string& key, const std::string& fallback) const
{
    const auto found = values.find(key);
    return found == values.end() ? fallback : found->second;
}
bool helpRequested(int argc, char** argv)
{
    return argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}
/* 先做选项结构校验，再由各入口和 parseCommon 校验数值/路径用途。
 * 不接受重复参数以免出现“最后一个值覆盖前一个值”的隐式配置。 */
bool parseArguments(int argc, char** argv, const std::set<std::string>& extraValues,
                    const std::set<std::string>& extraFlags,
                    Arguments& output, std::string& error)
{
    output = Arguments{};
    error.clear();
    const std::set<std::string> common{
        "--model", "--labels", "--num-classes", "--confidence", "--nms", "--gate"};
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (common.count(name) == 0 && extraValues.count(name) == 0
            && extraFlags.count(name) == 0) {
            error = "Unknown option: " + name;
            return false;
        }
        if (output.has(name)) {
            error = "Duplicate option: " + name;
            return false;
        }
        if (extraFlags.count(name) != 0) {
            output.flags.insert(name);
            continue;
        }
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0
            || std::string(argv[i + 1]).empty()) {
            error = "Missing or empty value for " + name;
            return false;
        }
        output.values.emplace(name, argv[++i]);
    }
    return true;
}
bool parseUnsigned(const std::string& text, std::uint64_t minimum,
                   std::uint64_t maximum, std::uint64_t& output)
{
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) {
            return c >= '0' && c <= '9';
        }))
        return false;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()
        || value < minimum || value > maximum)
        return false;
    output = value;
    return true;
}
bool parseFinite(const std::string& text, double minimum, double maximum,
                 bool minimumInclusive, double& output)
{
    if (text.empty() || std::any_of(text.begin(), text.end(), [](unsigned char c) {
            return std::isspace(c) != 0;
        }))
        return false;
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(value)
        || value > maximum || (minimumInclusive ? value < minimum : value <= minimum))
        return false;
    output = value;
    return true;
}
/* 类别数由用户声明或标签行数提供，不能从类别名字猜测模型输出通道含义。
 * 这里只核对参数之间的一致性；真正的张量布局/类数匹配由 detector.load 完成。 */
bool parseCommon(const Arguments& arguments, CommonOptions& output, std::string& error)
{
    output = CommonOptions{};
    error.clear();
    output.model = arguments.get("--model");
    if (output.model.empty()) {
        error = "--model is required.";
        return false;
    }
    if (!arguments.has("--labels") && !arguments.has("--num-classes")) {
        error = "Provide --labels or --num-classes; no built-in class names are assumed.";
        return false;
    }
    if (arguments.has("--labels") && !readLabels(arguments.get("--labels"), output.labels, error))
        return false;
    if (arguments.has("--num-classes")) {
        std::uint64_t count = 0;
        if (!parseUnsigned(arguments.get("--num-classes"), 1, 4096, count)) {
            error = "--num-classes must be an integer in [1,4096].";
            return false;
        }
        output.numClasses = static_cast<int>(count);
        if (!output.labels.empty() && output.labels.size() != count) {
            error = "--num-classes does not match the labels file line count.";
            return false;
        }
    } else
        output.numClasses = static_cast<int>(output.labels.size());
    /* NMS 下界不包含 0；还需防止正 double 转 float 后下溢到 0。 */
    const std::array<std::pair<const char*, float*>, 3> thresholds{{
        {"--confidence", &output.confidence}, {"--nms", &output.nms}, {"--gate", &output.gate}}};
    for (const auto& entry : thresholds) {
        if (!arguments.has(entry.first))
            continue;
        double threshold = 0;
        const bool inclusive = std::string(entry.first) != "--nms";
        if (!parseFinite(arguments.get(entry.first), 0, 1, inclusive, threshold)
            || (!inclusive && static_cast<float>(threshold) <= 0.0F)) {
            error = std::string(entry.first) + (inclusive
                ? " must be finite and in [0,1]." : " must be finite and in (0,1].");
            return false;
        }
        *entry.second = static_cast<float>(threshold);
    }
    return true;
}
void printCommonUsage(std::ostream& stream)
{
    stream << "  --model MODEL.rknn  --labels names.txt | --num-classes N\n"
              "  [--confidence 0.25] [--nms 0.45] [--gate 0.0]\n"
              "Class count: 1..4096. If both labels and count are given they must agree.\n"
              "Labels: one nonempty UTF-8 name per line; BOM/CRLF are accepted.\n"
              "No labels: class_<id>. UTF-8 names are printed in the console;\n"
              "non-ASCII overlay names use class_<id> (OpenCV's built-in font is ASCII).\n"
              "confidence/gate: [0,1]; nms: (0,1]. Unknown/duplicate options are errors.\n"
              "Outputs contain boxes and mask statistics, not full mask images.\n";
}
/* 打印实际加载模型的输入尺寸、类数和识别出的布局，便于和导出契约核对。
 * 标签用带引号文本输出，包含空格的类别名仍能区分边界。 */
void printModelInfo(std::ostream& stream, const YoloDetector& detector,
                    const CommonOptions& options)
{
    stream << "MODEL classes=" << detector.classCount()
           << " input=" << detector.inputWidth() << 'x' << detector.inputHeight()
           << " layout=" << layoutName(detector.outputLayout()) << '\n';
    for (std::size_t i = 0; i < options.labels.size(); ++i)
        stream << "LABEL id=" << i << " name=" << std::quoted(options.labels[i]) << '\n';
}
std::string classLabel(const CommonOptions& options, int classId)
{
    if (classId >= 0 && static_cast<std::size_t>(classId) < options.labels.size())
        return options.labels[static_cast<std::size_t>(classId)];
    return "class_" + std::to_string(classId);
}
/* 画面只展示检测几何：矩形框、中心圆点和底点十字。
 * OpenCV 内置字体无法绘制完整 Unicode，非 ASCII 名称在画面中改用类别编号，
 * 原始 UTF-8 标签仍由 printModelInfo/printDetections 输出到控制台。 */
void annotate(cv::Mat& image, const DetectionList& detections, const CommonOptions& options)
{
    for (const auto& d : detections) {
        const cv::Scalar color = classColor(d.classId);
        const cv::Point a(pixel(d.x1, image.cols), pixel(d.y1, image.rows));
        const cv::Point b(pixel(d.x2, image.cols), pixel(d.y2, image.rows));
        cv::rectangle(image, a, b, color, 2, cv::LINE_AA);
        cv::circle(image, cv::Point(pixel(d.cx, image.cols), pixel(d.cy, image.rows)),
                   4, color, cv::FILLED, cv::LINE_AA);
        cv::drawMarker(image, cv::Point(pixel(d.bottomX, image.cols), pixel(d.bottomY, image.rows)),
                       color, cv::MARKER_CROSS, 12, 2, cv::LINE_AA);
        std::string name = classLabel(options, d.classId);
        if (std::any_of(name.begin(), name.end(), [](unsigned char c) { return c < 0x20 || c > 0x7E; }))
            name = "class_" + std::to_string(d.classId);
        const std::string label = cv::format("%s %.2f", name.c_str(), d.score);
        int baseline = 0;
        const cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int y = std::min(image.rows - 1, std::max(size.height + 2, a.y - 4));
        cv::rectangle(image, cv::Point(a.x, std::max(0, y - size.height - 2)),
                      cv::Point(std::min(image.cols - 1, a.x + size.width + 3),
                                std::min(image.rows - 1, y + baseline + 2)),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(a.x + 1, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv::LINE_AA);
    }
}
/* bbox/center/bottom 均相对原图归一化；area 为面积占比，mask_valid 表示
 * 掩膜统计是否有效。没有逐像素掩膜数据，不能据此导出彩色实例分割蒙版。 */
void printDetections(std::ostream& stream, const DetectionList& detections,
                     const CommonOptions& options)
{
    stream << std::fixed << std::setprecision(6) << std::boolalpha;
    std::size_t index = 0;
    for (const auto& d : detections)
        stream << "detection=" << index++ << " class_id=" << d.classId
               << " class_name=" << std::quoted(classLabel(options, d.classId))
               << " score=" << d.score << " bbox_norm=[" << d.x1 << ',' << d.y1 << ','
               << d.x2 << ',' << d.y2 << "] box_area_norm=" << d.area
               << " mask_valid=" << d.maskValid << " mask_area_norm=" << d.maskArea
               << " center_norm=[" << d.cx << ',' << d.cy << "] bottom_norm=["
               << d.bottomX << ',' << d.bottomY << "] roi_mask_overlap=" << d.roiMaskOverlap << '\n';
}
} // namespace rk_yolo::example
