// SPDX-License-Identifier: Apache-2.0
/**
 * @file    camera_demo.cpp
 * @brief   V4L2 摄像头实时识别入口，默认无窗口，可选本地显示。
 *
 * --camera 支持设备编号或 /dev/ 路径；--width/--height 的单位为像素，
 * --fps 为请求帧率。--frames 0 持续运行，--log-every 按处理帧数控制日志频率。
 * 程序仅采集、识别和显示，不保存视频，不发送运动控制或图传数据。
 */
#include "image_io.hpp"
#include "shared_options.hpp"
#include <opencv2/videoio.hpp>
#if defined(RK_YOLO_WITH_HIGHGUI) && RK_YOLO_WITH_HIGHGUI
#include <opencv2/highgui.hpp>
#endif
#include <algorithm>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using rk_yolo::example::Arguments;
using rk_yolo::example::CommonOptions;
/* 信号处理只修改可用于异步信号通信的标志，不在处理函数里打印或释放资源。
 * 主循环恢复执行后检查标志；摄像头和可选窗口在正常栈展开中释放。 */
volatile std::sig_atomic_t stopped = 0;
void stopHandler(int) { stopped = 1; }

void usage(std::ostream& stream, const char* program)
{
    stream << "Usage: " << program << " --model MODEL.rknn (--labels names.txt | --num-classes N)\n"
              "       [--camera 0|/dev/video0] [--width 640] [--height 480] [--fps 30]\n"
              "       [--frames 0] [--log-every 30] [--display]\n";
    rk_yolo::example::printCommonUsage(stream);
    stream << "Uses OpenCV V4L2. Width/height/FPS are requests; actual negotiation is reported.\n"
              "width/height: even integers in [2,32768]; fps: (0,1000].\n"
              "frames: [0,4294967295], 0 means until Ctrl+C; log-every: [1,4294967295].\n"
              "Headless by default. --display requires -DYOLO_WITH_DISPLAY=ON at build time;\n"
              "press q or Esc in the window to stop. No video or frames are saved.\n"
              "loop_fps measures capture, conversion, inference, logging and optional drawing/display;\n"
              "it is separate from the model's preprocess/inference/postprocess timings.\n";
}

/* 请求参数：宽高为偶数像素，fps 为帧/秒；frames=0 表示不限帧数，
 * logEvery 至少为 1；display=false 时不创建窗口，也不做标注绘图。 */
struct CameraOptions final {
    std::string device{"0"};
    bool numeric{true};
    int index{0}, width{640}, height{480};
    double fps{30};
    std::uint64_t frames{0}, logEvery{30};
    bool display{false};
};
/**
 * @brief  校验摄像头专用参数，此阶段不打开设备。
 * @return true：参数可用于后续协商；false：error 给出原因。
 * @note   合法请求不代表驱动一定支持；实际图像尺寸会在采集后再次确认。
 *         无 highgui 构建拒绝 --display，公开构建开关为 YOLO_WITH_DISPLAY。
 */
bool parseCamera(const Arguments& arguments, CameraOptions& output, std::string& error)
{
    output.device = arguments.get("--camera", "0");
    output.numeric = !output.device.empty()
        && std::all_of(output.device.begin(), output.device.end(), [](unsigned char c) {
               return c >= '0' && c <= '9';
           });
    if (output.numeric) {
        std::uint64_t index = 0;
        if (!rk_yolo::example::parseUnsigned(output.device, 0,
                static_cast<std::uint64_t>(std::numeric_limits<int>::max()), index)) {
            error = "--camera numeric index is out of range.";
            return false;
        }
        output.index = static_cast<int>(index);
    } else if (output.device.rfind("/dev/", 0) != 0) {
        error = "--camera must be a nonnegative decimal index or a /dev/ device path.";
        return false;
    }
    std::uint64_t width = 0, height = 0;
    if (!rk_yolo::example::parseUnsigned(arguments.get("--width", "640"), 2, 32768, width)
        || !rk_yolo::example::parseUnsigned(arguments.get("--height", "480"), 2, 32768, height)
        || (width & 1U) || (height & 1U)) {
        error = "--width and --height must be even integers in [2,32768].";
        return false;
    }
    output.width = static_cast<int>(width);
    output.height = static_cast<int>(height);
    if (!rk_yolo::example::parseFinite(arguments.get("--fps", "30"), 0, 1000, false, output.fps)) {
        error = "--fps must be finite and in (0,1000].";
        return false;
    }
    if (!rk_yolo::example::parseUnsigned(arguments.get("--frames", "0"), 0, 4294967295ULL, output.frames)
        || !rk_yolo::example::parseUnsigned(arguments.get("--log-every", "30"), 1,
                                           4294967295ULL, output.logEvery)) {
        error = "--frames must be in [0,4294967295]; --log-every in [1,4294967295].";
        return false;
    }
    output.display = arguments.has("--display");
#if !defined(RK_YOLO_WITH_HIGHGUI) || !RK_YOLO_WITH_HIGHGUI
    if (output.display) {
        error = "--display is unavailable: rebuild with -DYOLO_WITH_DISPLAY=ON and OpenCV highgui.";
        return false;
    }
#endif
    return true;
}

#if defined(RK_YOLO_WITH_HIGHGUI) && RK_YOLO_WITH_HIGHGUI
/* 只在编译并启用显示时持有窗口；析构不把窗口清理异常继续向外抛出。 */
struct Window final {
    bool opened{false};
    ~Window() noexcept {
        if (opened)
            try { cv::destroyWindow("YOLO segmentation"); } catch (...) {}
    }
};
#endif

/**
 * @brief  加载模型、协商摄像头并按帧执行识别。
 * @return 0：达到帧数或用户请求退出；1：模型、取流、转换或推理失败。
 * @note   阶段耗时单位为 ms；循环 FPS 另以单调墙钟计量，包含实际启用的步骤。
 */
int run(const CommonOptions& options, const CameraOptions& camera)
{
    if (!std::filesystem::is_regular_file(options.model)) {
        std::cerr << "ERROR: Model file does not exist: " << options.model << '\n';
        return 1;
    }
    /* VideoCapture 默认输出 BGR，转换得到的 NV12 使用 BT.601 limited；
     * 此色彩范围描述的是转换后的输入，不是直接照搬摄像头原始格式的元数据。 */
    rk_yolo::YoloDetector detector;
    const auto loaded = detector.load(options.model, options.numClasses);
    if (!loaded) {
        std::cerr << "ERROR: Cannot load model: " << loaded.message << '\n';
        return 1;
    }
    if (!detector.setColorSpace(rk_yolo::CameraColorSpace::Bt601Limited)
        || !detector.setNmsIou(options.nms)) {
        std::cerr << "ERROR: Cannot configure detector.\n";
        return 1;
    }
    detector.setProfiling(true);
    rk_yolo::example::printModelInfo(std::cout, detector, options);
    /* 显式选择 V4L2，避免把设备路径当作其他输入源自动探测。 */
    cv::VideoCapture capture;
    const bool opened = camera.numeric ? capture.open(camera.index, cv::CAP_V4L2)
        : capture.open(camera.device, cv::CAP_V4L2);
    if (!opened) {
        if (stopped)
            return 0;
        std::cerr << "ERROR: Cannot open V4L2 camera: " << camera.device << '\n';
        return 1;
    }
    /* set 的返回值只记录请求是否被接口接受；驱动报告值与实际帧尺寸分别打印，
     * 不把设定成功当作目标分辨率/FPS 已生效，也不把驱动标称 FPS 当实测帧率。 */
    const bool widthSet = capture.set(cv::CAP_PROP_FRAME_WIDTH, camera.width);
    const bool heightSet = capture.set(cv::CAP_PROP_FRAME_HEIGHT, camera.height);
    const bool fpsSet = capture.set(cv::CAP_PROP_FPS, camera.fps);
    std::cout << std::fixed << std::setprecision(3) << std::boolalpha
              << "CAMERA requested_device=" << std::quoted(camera.device)
              << " requested_width=" << camera.width << " requested_height=" << camera.height
              << " requested_fps=" << camera.fps
              << " set_width_return=" << widthSet << " set_height_return=" << heightSet
              << " set_fps_return=" << fpsSet << '\n'
              << "CAMERA driver_reported_width=" << capture.get(cv::CAP_PROP_FRAME_WIDTH)
              << " driver_reported_height=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT)
              << " driver_reported_fps=" << capture.get(cv::CAP_PROP_FPS)
              << " display=" << camera.display << std::endl;
#if defined(RK_YOLO_WITH_HIGHGUI) && RK_YOLO_WITH_HIGHGUI
    Window window;
    if (camera.display) {
        cv::namedWindow("YOLO segmentation", cv::WINDOW_NORMAL);
        window.opened = true;
    }
#endif
    /* 计时从主循环开始，排除模型加载、设备打开和初始协商。
     * processed 仅统计完成识别的帧；truncatedFrames 统计被检测器标记为输出截断的帧。 */
    std::uint64_t processed = 0, windowFrames = 0, truncatedFrames = 0;
    int previousWidth = 0, previousHeight = 0;
    const auto start = Clock::now();
    auto windowStart = start;
    double preprocess = 0, inference = 0, postprocess = 0;
    /* 每轮：取 BGR 帧→转 NV12→同步推理→可选绘制/显示→按需打印日志。
     * 信号退出在取流返回后生效；若驱动阻塞，响应速度取决于底层读帧调用。 */
    while (!stopped && (camera.frames == 0 || processed < camera.frames)) {
        cv::Mat bgr;
        if (!capture.read(bgr) || bgr.empty()) {
            if (stopped)
                break;
            std::cerr << "ERROR: Camera read failed after " << processed << " frames.\n";
            return 1;
        }
        if (stopped)
            break;
        /* 以真正交付的图像确认尺寸；运行中尺寸变化也重新记录。 */
        if (bgr.cols != previousWidth || bgr.rows != previousHeight) {
            std::cout << "FRAME actual_width=" << bgr.cols << " actual_height=" << bgr.rows << std::endl;
            previousWidth = bgr.cols;
            previousHeight = bgr.rows;
        }
        /* 转换缓冲拥有完整帧；它的生命周期覆盖下面的同步 detect 调用。
         * 实际帧若为奇数尺寸、空帧或不支持的 BGR 类型，明确报错退出。 */
        rk_yolo::example::Nv12Image nv12;
        std::string error;
        if (!rk_yolo::example::bgrToNv12(bgr, nv12, error)) {
            std::cerr << "ERROR: Camera frame conversion: " << error << '\n';
            return 1;
        }
        const auto detections = detector.detect(nv12.view(), options.confidence, options.gate);
        if (!detections) {
            std::cerr << "ERROR: Inference failed: "
                      << rk_yolo::YoloDetector::errorName(detector.lastError())
                      << ": " << detections.status().message << '\n';
            return 1;
        }
        const auto timing = detector.timing();
        preprocess += timing.preprocessMilliseconds;
        inference += timing.inferenceMilliseconds;
        postprocess += timing.postprocessMilliseconds;
        truncatedFrames += detector.outputTruncated() ? 1U : 0U;
#if defined(RK_YOLO_WITH_HIGHGUI) && RK_YOLO_WITH_HIGHGUI
        /* 显示开销只在 --display 启用时发生；q/Esc 与 Ctrl+C 共用退出标志。 */
        if (camera.display) {
            rk_yolo::example::annotate(bgr, detections.value(), options);
            cv::imshow("YOLO segmentation", bgr);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q')
                stopped = 1;
        }
#endif
        ++processed;
        ++windowFrames;
        /* 首帧、每 logEvery 帧以及有限运行的末帧输出识别结果。
         * 阶段时间来自检测器的阶段计时；loop_fps 为该日志窗口的循环速率，累计值包含先前
         * 的取流、转换、推理、可选显示及日志，不用阶段耗时倒数冒充整机 FPS。 */
        if (processed == 1 || processed % camera.logEvery == 0
            || (camera.frames != 0 && processed == camera.frames)) {
            const auto now = Clock::now();
            const double windowSeconds = std::chrono::duration<double>(now - windowStart).count();
            const double totalSeconds = std::chrono::duration<double>(now - start).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "FRAME index=" << processed << " detections=" << detections.value().size()
                      << " truncated=" << detector.outputTruncated()
                      << " preprocess_ms=" << timing.preprocessMilliseconds
                      << " inference_ms=" << timing.inferenceMilliseconds
                      << " postprocess_ms=" << timing.postprocessMilliseconds
                      << " loop_fps=" << (windowSeconds > 0 ? windowFrames / windowSeconds : 0)
                      << " cumulative_loop_fps=" << (totalSeconds > 0 ? processed / totalSeconds : 0)
                      << std::endl;
            rk_yolo::example::printDetections(std::cout, detections.value(), options);
            std::cout.flush();
            windowFrames = 0;
            windowStart = now; // 下一计时窗口包含本次日志输出的耗时。
        }
    }
    /* 总结按整个循环墙钟计算，包括最后一次日志；0 帧退出时避免除零。 */
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const double count = static_cast<double>(processed);
    std::cout << std::fixed << std::setprecision(3)
              << "CAMERA_DONE frames=" << processed << " truncated_frames=" << truncatedFrames
              << " elapsed_seconds=" << seconds << " loop_fps=" << (seconds > 0 ? count / seconds : 0)
              << " avg_preprocess_ms=" << (processed ? preprocess / count : 0)
              << " avg_inference_ms=" << (processed ? inference / count : 0)
              << " avg_postprocess_ms=" << (processed ? postprocess / count : 0)
              << " stopped_by_signal_or_key=" << (stopped != 0) << std::endl;
    return 0;
}
} // namespace

/* 参数/显示能力检查在访问设备之前完成；SIGINT/SIGTERM 仅通知主循环退出。 */
int main(int argc, char** argv)
{
    try {
        if (rk_yolo::example::helpRequested(argc, argv)) {
            usage(std::cout, argv[0]);
            return 0;
        }
        Arguments arguments;
        CommonOptions options;
        CameraOptions camera;
        std::string error;
        if (!rk_yolo::example::parseArguments(argc, argv,
                {"--camera", "--width", "--height", "--fps", "--frames", "--log-every"},
                {"--display"}, arguments, error)
            || !rk_yolo::example::parseCommon(arguments, options, error)
            || !parseCamera(arguments, camera, error)) {
            std::cerr << "ERROR: " << error << '\n';
            usage(std::cerr, argv[0]);
            return 2;
        }
        std::signal(SIGINT, stopHandler);
        std::signal(SIGTERM, stopHandler);
        return run(options, camera);
    } catch (const cv::Exception& error) {
        if (stopped)
            return 0;
        std::cerr << "ERROR: OpenCV: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "ERROR: Unknown failure.\n";
    }
    return 1;
}
