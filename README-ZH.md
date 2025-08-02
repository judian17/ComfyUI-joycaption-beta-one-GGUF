# ComfyUI JoyCaption-Beta-GGUF Node

本项目是 ComfyUI 的一个节点，用于使用 GGUF 格式的 JoyCaption-Beta 模型进行图像描述。

**致谢:**

本项目基于 [fpgaminer/joycaption_comfyui](https://github.com/fpgaminer/joycaption_comfyui) 进行修改，主要变化在于支持 GGUF 模型格式。

感谢[layerstyleadvance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance)节点，我从中复制了extra options相关代码

**20250802-更新：**
由于安装gpu版本的llama-cpp-python比较困难，我将模型上传至ollama，现在可以通过安装[ollama](https://ollama.com/)与[comfyui-ollama](https://github.com/stavsap/comfyui-ollama)来使用。模型链接：https://ollama.com/aha2025/llama-joycaption-beta-one-hf-llava  同时我将提示词模板分离为一个单独的节点，详情可查看[ComfyUI JoyCaption-Beta-GGUF](https://github.com/judian17/ComfyUI-JoyCaption-beta-one-hf-llava-Prompt_node) 节点

## 使用方法

### 安装依赖

本节点需要安装 `llama-cpp-python`。

**重要提示:**

* 直接使用 `pip install llama-cpp-python` 安装只能在 CPU 上运行。
* 如需使用 NVIDIA GPU 加速推理，请使用以下命令安装：
    ```bash
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    ```
    *(请根据您的 CUDA 版本调整 `cu124`)*
* 非英伟达显卡或其他安装方法，请参考 `llama-cpp-python` 官方文档：
    [https://llama-cpp-python.readthedocs.io/en/latest/](https://llama-cpp-python.readthedocs.io/en/latest/)

`llama-cpp-python` 未在 `requirements.txt` 中列出，请手动安装以确保选择正确的 GPU 支持版本。

### 工作流示例

您可以在 `assets/example.png` 查看工作流示例图。

![工作流示例](assets/example.png)

### 模型下载与放置

您需要下载 JoyCaption-Beta 的 GGUF 模型和相关的 mmproj 模型。

1.  从以下 Hugging Face 仓库下载模型：
    * **主模型 (推荐):** [concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf](https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main)
        * 下载对应的 `joycaption-beta` 模型文件和 `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf` 文件。
    * **其他量化版本:** [mradermacher/llama-joycaption-beta-one-hf-llava-GGUF](https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/tree/main)
    * **IQ 量化版本 (理论上质量更高，CPU 推理可能较慢):** [mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF](https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF/tree/main)

2.  将下载的模型文件放置到您的 ComfyUI 安装目录下的 `models\llava_gguf\` 文件夹内。

### 视频教程

您可以参考以下 Bilibili 视频教程进行设置和使用：

[视频](https://www.bilibili.com/video/BV1JKJgzZEgR/)
