# 🧠 常用视觉模型的 ONNX 与 OpenVINO 推理实现  
> **Document the ONNX and OpenVINO inference for some commonly used vision models**

本项目记录并实现了多个常用视觉模型（如目标检测、图像分类、OCR 等）在 ONNX 格式下的推理流程，并进一步转换为 OpenVINO 格式以加速推理。

---

## 📦 支持的模型类型

目前支持以下模型类别及其推理实现：

| 模型类别       | 模型名称示例              | ONNX 推理 | OpenVINO 推理 |
|----------------|---------------------------|-----------|----------------|
| OCR 文字检测   | paddleOCR v5 mobile         | ✅         | ✅              |
| OCR 文字识别   | paddleOCR v5 mobile          | ✅         | ✅              |


更多模型正在持续更新中...

---

## 🛠️ 安装依赖

Python>=3.10


### 使用 pip 安装

```bash
pip install -r requirements.txt
```

---
## 📦 模型下载

### 方法一：使用我们提供的预转换模型（推荐）

我们提供了一些常见视觉模型的 ONNX 及 OpenVINO IR 格式的预转换模型，可通过以下链接下载：

🔗 [Google Drive 下载地址](#)

🔗 [百度网盘下载地址,提取码：fxw6](https://pan.baidu.com/s/1z6jj0o0fgtqbV6nfUt8VGw?pwd=fxw6)

解压后将模型文件夹放入 `weights/` 目录下


### 方法二：自行转换 ONNX 模型为 OpenVINO IR 格式


---

## 🔧 使用方法(示例)

1. OCR检测示例
```bash
cd examples
python test_ocr_det.py 
```

2. OCR识别示例
```bash
cd examples
python test_ocr_recog.py 
```

其余的模型使用方式都可参考examples中的样例

---

## 🚀 性能对比（示例）

测试10次取平均值


| 模型          | 平台       | 推理时间 (ms) | 
|---------------|------------|----------------|
| ppocr-v5-det-mobile      | ONNX CPU   |             |   
| ppocr-v5-det-mobile      | ONNX GPU   |             |   
| ppocr-v5-det-mobile     | OpenVINO CPU |             |   
| ppocr-v5-recog-mobile      | ONNX CPU   |              |    
| ppocr-v5-recog-mobile      | ONNX GPU   |             |    
| ppocr-v5-recog-mobile     | OpenVINO CPU |           |   


> 注：以上数据仅为示例，具体性能视硬件配置而定。

---
## 🌟 致谢

感谢以下开源项目的支持与启发：

- [ONNX](https://onnx.ai/)
- [OpenVINO™ Toolkit](https://docs.openvino.ai/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

