# YOLOv11n Free Diving Detection

使用 Ultralytics YOLOv11 nano 模型进行自由潜水游泳者检测

## 项目简介

本项目将目前最轻量的 YOLOv11n 模型应用于自由潜水（Freediving）场景下的游泳者实时检测。

目标是在水下、半水面、泳池、开放水域等多种复杂环境中，快速、稳定地检测出正在进行自由潜水的潜水员（包括下潜、悬停、上浮、换气等状态）。

主要特点：
- 极致轻量：YOLOv11n 参数量约 2.6M，适合边缘设备（Jetson、Raspberry Pi、移动端等）
- 实时性优秀：多数设备上可轻松达到 40~100+ FPS
- 针对水下特殊环境优化：光线折射、反光、水纹、模糊、姿态多变、部分遮挡等

当前主要检测类别（可根据你的数据集调整）：
- `freediver` —— 完整自由潜水者身体
- `freediver_head` —— 自由潜水者头部（常用于水面换气/呼吸检测）

## 目前性能（示例数据）

| 模型版本       | mAP@0.5 | mAP@0.5:0.95 | 参数量  | GFLOPs | FPS (RTX 4060 Laptop) | FPS (Jetson Orin Nano) |
|----------------|---------|--------------|---------|--------|------------------------|------------------------|
| YOLOv11n-base  | 91.2%   | 67.8%        | ~2.6M   | 6.5    | ~220fps                | ~45–55fps              |
| 本项目微调版   | **94.7%** | **71.9%**  | ~2.6M   | 6.5    | ~210–225fps            | **~50–65fps**          |

*测试条件：640×640 输入，TensorRT fp16 优化，batch=1*

## 快速开始

### 1. 环境准备

```bash
# 推荐使用 conda 新环境
conda create -n yolo11 python=3.11 -y
conda activate yolo11

# 安装最新 ultralytics（建议使用最新版）
pip install -U ultralytics
```

### 2. 直接推理演示
```Python
from ultralytics import YOLO

# 加载模型（请替换成你自己的权重路径）
model = YOLO("weights/best.pt")

# 单张图片推理
results = model("demo/diver_sample.jpg", conf=0.32, iou=0.45, imgsz=640)

# 视频/摄像头实时推理并保存结果
model.predict(
    source="demo/underwater_clip.mp4",    # 也可填 0 使用摄像头
    show=True,
    save=True,
    conf=0.30,
    iou=0.48,
    imgsz=640,
    half=True                      # 开启半精度加速（推荐）
)
```

### 3. 训练参考命令
```Bash
# 单卡训练
yolo detect train model=yolo11n.pt data=dataset.yaml epochs=100 imgsz=640 batch=32 device=0

# 多卡训练示例（4卡）
yolo detect train model=yolo11n.pt data=dataset.yaml epochs=120 imgsz=640 batch=128 workers=16 device=0,1,2,3
```

项目结构（推荐组织方式）
```text
YOLO-v11n-free-diving-detection/
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── weights/
│   └── best.pt               ← 训练得到的最佳权重
├── demo/
│   ├── images/
│   └── videos/
├── runs/                     ← 训练/验证/预测结果自动保存位置
├── README.md
└── requirements.txt 
```
## 未来计划
 - 增加更多类别（浮潜、救生员、水下摄影师等）
 - 结合关键点/姿态估计，判断“黑屏”（breath-hold blackout）危险状态
 - 开发移动端实时演示（Android/iOS）
 - 尝试 YOLO11s/m 更大模型做精度上限对比
 - 加入目标跟踪功能（ByteTrack/BoT-SORT）
