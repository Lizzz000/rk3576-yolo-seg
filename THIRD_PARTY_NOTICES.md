# 来源与许可

提取并通用化既有 RK3576 算法层；自编源码、构建和文档采用 [Apache-2.0](LICENSE)。

DFL 解码保留 Rockchip `rknn_model_zoo` 的 `compute_dfl` 来源注释（[Apache-2.0](https://github.com/airockchip/rknn_model_zoo/blob/main/LICENSE)）；其许可不覆盖模型或 SDK。

外部依赖未打包，须保留各实际版本许可：

| 组件 | 获取与许可说明 |
| --- | --- |
| RKNN runtime / SDK | [RKNN](https://github.com/airockchip/rknn-toolkit2)、[许可](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE)；不重新许可 |
| RGA | [librga](https://github.com/airockchip/librga)；头文件、库和驱动须配套 |
| OpenCV | [许可](https://opencv.org/license/)，以安装版本为准 |
| YOLO 模型与权重 | 自备；本项目许可不覆盖模型、基础权重、训练框架和数据集 |

Ultralytics 模型默认 AGPL-3.0，另有 Enterprise License，见 [官方说明](https://www.ultralytics.com/license)。其训练、导出条款须核对；C++ 推理或 RKNN 转换不自动免除原许可。

公开模型、样本和校准图片须确认来源及权限。
