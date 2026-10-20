# 模型契约

模型自备，仅适配以下分割输出；形状校验不保证坐标和类别语义正确。

## 输入与通用约定

- 单个 RGB 输入：NHWC `[1,H,W,3]` 或 NCHW `[1,3,H,W]`，宽高须能被 4 整除；实际送入 NHWC UINT8 RGB。
- 等比例双线性缩放，letterbox 填充 RGB 114；BGR→NV12 使用 BT.601 limited，须与 `setColorSpace()` 一致。
- `C` 为 `--num-classes` 或标签行数，范围 1..4096；类别分数须为 0..1，不再 sigmoid，无 objectness。
- 固定 32 个 mask 通道，prototype 为 NCHW `[1,32,H/4,W/4]`。

## 支持的输出

| 布局 | 张量约定 | 框语义 |
| --- | --- | --- |
| Packed | `[1,4+C+32,N]` + prototype | 模型像素 `cx,cy,w,h` + C 个分数 + 32 个系数 |
| Raw split | `[1,4,N]` 框、`[1,C,N]` 分数、`[1,32,N]` 系数、prototype | 模型像素 `cx,cy,w,h` |
| Split DFL | stride 8/16/32 的框分布、分数、可选 gate、系数 + prototype | 64 通道，reg_max=16 |
| End-to-end | `[1,N,38]` + prototype | 模型像素 `x1,y1,x2,y2,score,class_id` + 32 个系数 |

Packed 不支持转置布局 `[1,N,4+C+32]`。所有路径使用类别内 NMS，保留 0..C-1 原始类别编号，无重映射或过滤。

## 同形输出与输出顺序

- Raw split 默认顺序：`boxes, class_scores, mask_coefficients, prototype`，C=4/32 时尤其不能交换同形输出。
- Split DFL 默认顺序：stride 8、16、32；每尺度为 `boxes, class_scores, [gate], mask_coefficients`，最后为 prototype，共 13 或 10 个输出。同尺度空间尺寸须一致，C=1/32/64 时尤其要核对顺序。
- 明确的角色名可辅助识别，DFL 可带 `_s8/_s16/_s32` 后缀；名称未知且形状冲突时须遵循完整顺序。数量、维度、stride 不完整或仍有歧义时拒绝加载。

## 数值与容量

所用张量全部为 affine INT8 或全部为 FP16 时，分别走原生后处理；其余支持类型由 RKNN 转为 float。End-to-end INT8 若让坐标和分数共用量化尺度，可能损失置信度精度。

最多返回 64 个结果，须检查 `outputTruncated()`。框、中心、底点相对原图归一化；掩膜仅提供低分辨率统计，无完整像素、多边形或物理测量含义。

## 准备与替换模型

针对 RK3576 转换并匹配 runtime、驱动；保存工具版本、类别、输入输出与量化配置。换模型后先用固定图片核对结果，再测性能。参考 [Rockchip 示例](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8_seg)。

不含训练、量化或权重下载脚本，不保证支持其他检测、姿态、旋转框或分类模型。
