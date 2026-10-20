# 性能说明

选取原项目相机测试中最高观测帧率的配置，作为**历史实测参考**。当前通用版尚未重新上板实测。

| 项目 | 测试条件与结果 |
| --- | --- |
| 开发板 | 嘉立创泰山派 RK3576 开发板 |
| 测试日期 | 2026-09-04 |
| 模型 | YOLOv8n-seg INT8，4 类，输入 1280×736（宽×高），13 路 split DFL 输出 |
| 相机 | CSI/RKISP，1920×1080 NV12，采集约 30 FPS |
| NPU 配置 | `CORE_0_1` |
| 视觉处理 FPS | 约 **22.0–24.1**，最高观测约 **24.1** |
| 场景 | 原工程视觉链路，画面没有稳定目标 |

24.1 FPS 为最高观测值，不代表持续平均速度或多目标性能。通用版采用 OpenCV 取流和 BGR→NV12 转换，运行负载有所变化，需独立测量。

来源：原项目 `RK3576/docs/validation-2026-09-04-yolov8nseg.md` 的 NPU 核心配置测试记录。

## 通用版复测

```sh
./build/yolo_seg_image --model model.rknn --labels labels.txt \
  --image input.jpg --output result.jpg --warmup 10 --iterations 100
./build/yolo_seg_camera --model model.rknn --labels labels.txt \
  --camera /dev/video0 --width 640 --height 480 --fps 30 --frames 300
```

- `preprocess_ms`：RGA 转换与 letterbox；`inference_ms`：RKNN 输入、运行和取得输出；`postprocess_ms`：解码、NMS 与掩膜统计。
- 图片 `detect_wall_fps` 不含读写图和 BGR→NV12；相机 `loop_fps` 包含取流、转换和循环内处理。FPS 不等于完整显示延迟。
- 对比时固定模型、输入、目标数量、SDK、显示及日志设置，并保持相同的供电、散热与调频条件。
