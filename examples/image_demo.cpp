// SPDX-License-Identifier: Apache-2.0
/**
 * @file    image_demo.cpp
 * @brief   单张图片识别入口：参数校验、模型加载、重复推理与标注图片保存。
 *
 * 使用 --model 指定 RKNN 模型，--image 指定 JPG/PNG；类别由 --labels
 * 或 --num-classes 显式提供。--warmup 控制预热次数，--iterations 控制
 * 正式测量次数；最终保存最后一次检测的框、类别、中心与底点，不导出完整掩膜。
 */
#include "image_io.hpp"
#include "shared_options.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using rk_yolo::example::Arguments;
using rk_yolo::example::CommonOptions;

/* 图片入口的帮助信息。通用类别与阈值用法由共享模块统一补充。 */
void usage(std::ostream& stream, const char* program)
{
    stream << "Usage: " << program << " --model MODEL.rknn --image IMAGE.jpg|IMAGE.png\n"
              "       (--labels names.txt | --num-classes N)\n"
              "       [--output result.jpg] [--warmup 0] [--iterations 1]\n";
    rk_yolo::example::printCommonUsage(stream);
    stream << "Image width/height must be even for NV12; source resolution is preserved.\n"
              "warmup: [0,1000000]; iterations: [1,1000000]. No window is opened.\n"
              "detect_wall_fps measures repeated detect() calls after warmup; it excludes\n"
              "image decode, BGR-to-NV12 conversion, drawing and file saving.\n";
}
/* 同时比较规范化绝对路径和已存在文件的身份，避免输出覆盖输入文件或其链接。 */
bool sameFile(const std::filesystem::path& a, const std::filesystem::path& b)
{
    if (std::filesystem::absolute(a).lexically_normal() == std::filesystem::absolute(b).lexically_normal())
        return true;
    std::error_code error;
    return std::filesystem::equivalent(a, b, error) && !error;
}
/* 扩展名仅用于限定示例输入类型；文件能否解码仍由后续 imread 判断。 */
bool supportedImage(const std::string& path)
{
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}
/**
 * @brief  执行图片识别并保存最后一轮标注结果。
 * @param  arguments 已完成选项名与通用参数校验的命令行参数。
 * @param  options   模型路径、类别数/标签和检测阈值。
 * @return 0：图片保存完成；1：文件、转换或推理失败；2：图片专用参数无效。
 */
int run(const Arguments& arguments, const CommonOptions& options)
{
    const std::string imagePath = arguments.get("--image");
    const std::string outputPath = arguments.get("--output", "result.jpg");
    if (imagePath.empty()) {
        std::cerr << "ERROR: --image is required.\n";
        return 2;
    }
    /* 次数均以“完整 detect 调用”为单位；正式轮数至少为 1，保证均值分母有效。 */
    std::uint64_t warmup = 0, iterations = 1;
    if (!rk_yolo::example::parseUnsigned(arguments.get("--warmup", "0"), 0, 1000000, warmup)
        || !rk_yolo::example::parseUnsigned(arguments.get("--iterations", "1"), 1, 1000000, iterations)) {
        std::cerr << "ERROR: warmup must be in [0,1000000]; iterations in [1,1000000].\n";
        return 2;
    }
    if (!std::filesystem::is_regular_file(options.model)
        || !std::filesystem::is_regular_file(imagePath) || !supportedImage(imagePath)) {
        std::cerr << "ERROR: Expected an existing model and a JPG or PNG input image.\n";
        return 1;
    }
    if (sameFile(imagePath, outputPath) || sameFile(options.model, outputPath)
        || (arguments.has("--labels") && sameFile(arguments.get("--labels"), outputPath))) {
        std::cerr << "ERROR: --output must differ from every input file.\n";
        return 2;
    }
    /* 读图和 BGR→NV12 只执行一次，不计入后面的重复推理墙钟时间。
     * 保留原图尺寸，偶数尺寸约束和 BT.601 limited 色彩范围由转换模块负责。 */
    const cv::Mat source = cv::imread(imagePath, cv::IMREAD_COLOR);
    rk_yolo::example::Nv12Image nv12;
    std::string error;
    if (!rk_yolo::example::bgrToNv12(source, nv12, error)) {
        std::cerr << "ERROR: " << error << '\n';
        return 1;
    }
    /* 类别数参与模型输出契约检查，不能用任意标签数量解释不匹配的模型。
     * 加载后再设置色彩范围和 NMS；nv12 的存储在整个同步推理期间保持有效。 */
    rk_yolo::YoloDetector detector;
    const auto loaded = detector.load(options.model, options.numClasses);
    if (!loaded) {
        std::cerr << "ERROR: Cannot load model: " << loaded.message << '\n';
        return 1;
    }
    if (!detector.setColorSpace(nv12.colorSpace) || !detector.setNmsIou(options.nms)) {
        std::cerr << "ERROR: Cannot configure detector.\n";
        return 1;
    }
    detector.setProfiling(true);
    rk_yolo::example::printModelInfo(std::cout, detector, options);
    /* 阶段时间单位为 ms。预热轮不累计；保存的结果来自最后一轮正式推理。 */
    rk_yolo::DetectionList finalDetections;
    double preprocess = 0, inference = 0, postprocess = 0;
    std::uint64_t truncatedIterations = 0;
    auto measuredStart = Clock::now();
    /* 在预热与正式轮的边界重置单调时钟。detect_wall_fps 反映重复调用吞吐，
     * 包含 detect 本身及循环记账，不包含读图、NV12 转换、绘图和文件写入，
     * 因此不能直接当作摄像头端到端帧率，也不是单独的 NPU 内核吞吐。 */
    for (std::uint64_t i = 0; i < warmup + iterations; ++i) {
        if (i == warmup)
            measuredStart = Clock::now();
        const auto result = detector.detect(nv12.view(), options.confidence, options.gate);
        if (!result) {
            std::cerr << "ERROR: Inference failed at iteration " << i << ": "
                      << rk_yolo::YoloDetector::errorName(detector.lastError())
                      << ": " << result.status().message << '\n';
            return 1;
        }
        if (i >= warmup) {
            const auto timing = detector.timing();
            preprocess += timing.preprocessMilliseconds;
            inference += timing.inferenceMilliseconds;
            postprocess += timing.postprocessMilliseconds;
            truncatedIterations += detector.outputTruncated() ? 1U : 0U;
            finalDetections = result.value();
        }
    }
    const double elapsedMs = std::chrono::duration<double, std::milli>(Clock::now() - measuredStart).count();
    /* 在原图副本上绘制，保持原输入不变。写图返回成功且文件非空后，
     * 才输出 SUCCESS；空检测也是正常结果，会保存没有检测标记的原图。 */
    cv::Mat annotated = source.clone();
    rk_yolo::example::annotate(annotated, finalDetections, options);
    if (!cv::imwrite(outputPath, annotated) || !std::filesystem::is_regular_file(outputPath)
        || std::filesystem::file_size(outputPath) == 0) {
        std::cerr << "ERROR: Cannot save annotated image: " << outputPath << '\n';
        return 1;
    }
    rk_yolo::example::printDetections(std::cout, finalDetections, options);
    const double count = static_cast<double>(iterations);
    std::cout << std::fixed << std::setprecision(3) << std::boolalpha
              << "SUCCESS detections=" << finalDetections.size()
              << " truncated=" << detector.outputTruncated()
              << " truncated_iterations=" << truncatedIterations << " warmup=" << warmup
              << " iterations=" << iterations
              << " avg_preprocess_ms=" << preprocess / count
              << " avg_inference_ms=" << inference / count
              << " avg_postprocess_ms=" << postprocess / count
              << " detect_wall_ms=" << elapsedMs
              << " detect_wall_fps=" << (elapsedMs > 0 ? count * 1000.0 / elapsedMs : 0)
              << " output=" << std::quoted(outputPath) << '\n';
    return 0;
}
} // namespace

/* 参数错误返回 2，运行失败/异常返回 1；--help 和正常保存返回 0。 */
int main(int argc, char** argv)
{
    try {
        if (rk_yolo::example::helpRequested(argc, argv)) {
            usage(std::cout, argv[0]);
            return 0;
        }
        Arguments arguments;
        CommonOptions options;
        std::string error;
        if (!rk_yolo::example::parseArguments(argc, argv,
                {"--image", "--output", "--warmup", "--iterations"}, {}, arguments, error)
            || !rk_yolo::example::parseCommon(arguments, options, error)) {
            std::cerr << "ERROR: " << error << '\n';
            usage(std::cerr, argv[0]);
            return 2;
        }
        return run(arguments, options);
    } catch (const cv::Exception& error) {
        std::cerr << "ERROR: OpenCV: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "ERROR: Unknown failure.\n";
    }
    return 1;
}
