# NLP_Final_Work
深圳大学NLP期末大作业代码与结果库


本仓库包含了一个Vision Transformer (ViT) 模型的代码和实验结果，该模型应用于多种计算机视觉任务。仓库中涵盖的任务包括：

- **Multiple Classification**：图像多分类任务。
- **Object Detection**：目标检测任务，旨在检测图像中的物体并进行定位。
- **Semantic Segmentation**：语义分割任务，将图像划分为具有语义意义的不同部分。

## 目录结构
.
├── classification/
│   ├── model.py       # 用于分类任务的ViT模型
│   ├── train.py       # 分类任务的训练脚本
│   ├── evaluate.py    # 分类任务的评估脚本
│   └── results/       # 存储分类任务结果的文件夹
├── detection/
│   ├── test.py
│   ├── train.py       # 检测任务的训练脚
│   └── results/       # 存储检测任务结果的文件夹
├── segmentation/
│   ├── model.py       # 用于分割任务的ViT模型
│   ├── train.py       # 分割任务的训练脚本
│   ├── evaluate.py    # 分割任务的评估脚本
│   └── results/       # 存储分割任务结果的文件夹
├── README.md          # 本文件
└── requirements.txt   # 所需的Python依赖包


## 环境要求

为了运行代码和进行实验，您需要安装以下依赖：

- Python 3.x
- PyTorch
- TensorFlow（用于某些分割任务）
- TensorBoard
- numpy
- matplotlib


运行代码

1. 多分类任务 (Multiple Classification)

要训练用于图像分类的ViT模型，可以运行以下命令：
    python classification/train.py



## 许可证

本项目采用MIT许可证 - 详情请查看 LICENSE 文件。

### 致谢
	•	Vision Transformer (ViT) 模型的实现参考了 ViT Paper。
	•	感谢所有贡献者和开源库，它们使得本项目得以顺利完成。
