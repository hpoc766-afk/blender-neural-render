# Neural Render Bridge

[中文](#中文) · [English](#english)

[GitHub](https://github.com/hpoc766-afk/neural-render-bridge) · [Download / 下载](https://github.com/hpoc766-afk/neural-render-bridge/releases/latest)

## 中文

### 功能与处理位置

Neural Render Bridge 是 Blender 5.1 的单帧神经渲染插件。它在**渲染器输出之后、原合成器后处理之前**读取同帧场景线性 HDR，用 PyTorch 处理颜色，再将处理后的 HDR RGB 交回原合成图。原有 AOV、遮罩、alpha、合成节点和显示变换继续参与最终输出。

网络输入来自 Render Layers 的 Combined 缓冲；最终 PNG 仅用于查看结果，不会作为网络输入。

```text
当前场景快照 → 原生渲染一次 → 同帧 Combined HDR + 原合成图使用的通道
                                  ├─ 原颜色 ──────────→ 原合成器 → Original
                                  ├─ 身份颜色 ────────→ 原合成器 → Identity
                                  └─ PyTorch → HDR RGB → 原合成器 → Neural
```

### 安装与依赖

1. 在 Blender 中打开 **Edit → Preferences → Get Extensions**，从菜单选择 **Install from Disk**，安装 `neural_render_bridge-0.1.0.zip` 并启用插件。
2. 准备独立的 Python 3.10 或更高版本，安装匹配显卡和驱动的 CUDA PyTorch，以及 `numpy`、`Pillow`、`OpenEXR`。不要把这些依赖装入 Blender 自带的 Python。
3. 准备模型源码目录与兼容的 `dlss5_static.pt` 权重。模型源码目录必须直接包含 `dlss5/graph.py`，例如模型仓库中的 `src` 目录。模型源码上游为 [taowen/dlss5-onnx](https://github.com/taowen/dlss5-onnx)。
4. 在插件偏好设置中配置 **Python executable**、**Model source directory**、**Checkpoint**，以及可选的 **Output directory**。
5. 点击 **Check Environment**。检查会确认模块可以导入；神经模式还会检查 CUDA 可用性及权重文件是否存在。权重的结构兼容性会在实际加载时验证。

安装包包含 Blender UI、任务管理器与渲染桥接实现。PyTorch、模型源码、模型权重和引擎二进制由本机提供，不包含在安装包中。身份验证模式和强度为零的模式不加载权重，但仍需要可以导入的 Python 依赖与模型源码。

外部 Python 环境的依赖可用以下命令安装；CUDA PyTorch 请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择版本。

```bash
python -m pip install numpy Pillow OpenEXR
```

### 使用

在 **Properties → Render → Neural Render Bridge** 中选择模式，点击 **Render with Neural Pipeline**。

| 设置 | 含义 |
|---|---|
| Neural | 执行 PyTorch 推理并回放原合成器 |
| Identity validation | 保持 HDR 不变，检查捕获和原合成器回放 |
| Strength | HDR 残差强度，范围 0–1；0 跳过模型推理 |
| Eager inference | 使用普通 PyTorch 前向代替 CUDA Graph |

插件使用当前场景、当前整数帧，以及原合成图 Render Layers 节点指定的渲染层。它将当前内存场景及依赖写入独立 `.blend` 快照，并将外部路径转换为绝对路径；原打开文件不会被覆盖。文件纹理有未保存的像素编辑时，需要先保存或打包纹理。

任务在后台执行，界面显示当前阶段。**Cancel Job** 仅终止该任务启动的工作进程及子进程。完成后可通过 **View Original / View Identity / View Neural** 加载对应图像，或通过 **Open Output Folder** 查看完整记录。

请使用插件的渲染按钮。普通 F12 保持 Blender 的原生行为；此插件没有注册替代 RenderEngine，也没有更改 F12 快捷键。最终结果保存到任务目录，并可加载到 Image Editor。

### 实现方式

**进程分工。** Blender 主进程只负责偏好设置、场景快照、任务状态与图像查看。独立 Python 工作进程加载模型，并启动同一个 Blender 可执行文件的后台实例执行原生渲染和合成回放。PyTorch 不在 Blender 主进程中运行。

**捕获边界。** 桥接模块在后台场景的原 Render Layers 输出上增加 File Output 捕获节点，将 Combined 和原合成图消费的辅助通道保存为 FLOAT32、无损 ZIP 压缩的多层 OpenEXR。数据属于同一次源渲染。原合成器最终输入也在该次渲染中导出，作为独立原生参考。合成器本身无需读取先前保存的 PNG。

**模型适配。** 适配器从原 HDR RGB 中建立非负模型输入，执行曝光缩放与 Reinhard 映射，再等比例缩放并复制边缘补齐至 256×256。公开静态网络使用 NCHW、FP16 和可选 CUDA Graph。输出去掉补边并恢复原分辨率。静态前端以当前颜色替代历史颜色，未知标量与辅助特征保留公开实现的默认值；它不消费场景运动向量或深度来恢复原生时序接口。

**HDR 回写。** 对模型输入和输出分别做有界逆 Reinhard，二者差值作为残差锚定到原始 HDR：

```text
restored_HDR = original_HDR + strength × (inverse(output) − inverse(input))
```

逆映射限制在适配器定义的 HDR 范围内，原始 HDR 基底不裁切，因此零残差能保留负值和高亮值。alpha 沿用原始捕获值；辅助通道不经过模型。

**原后处理回接。** 回放阶段将 Render Layers 节点替换为同帧多层 EXR Image 节点，并仅对原先来自 `Image` 的 RGB 连接替换为网络 HDR。深度、法线、AOV 和遮罩沿用缓存。专用回放 RenderEngine 禁止几何渲染，只执行原合成图；原有颜色管理负责最终显示编码。

**验收与缓存绑定。** 捕获清单记录场景快照、帧、相机、渲染层、尺寸、设置、外部依赖、源代码摘要与资源摘要。分支输入绑定捕获 ID 和 alpha 摘要。直通与身份分支必须同时通过浮点和 PNG 原生参考比较，PNG 最大允许差为 1 个 8-bit 编码值，浮点最大允许差为 `1e-6`。不支持的结构或比较失败会停止任务，并保存错误及阶段日志。

### 支持范围与输出

已验证环境为 Windows、Blender 5.1.2、CUDA PyTorch 与 NVIDIA GPU。插件要求 Blender 5.1；神经推理需要兼容 CUDA 的 GPU。当前支持单帧、标准完整画幅、Linear Rec.709、启用的原合成图及恰好一个本场景 Render Layers 源。渲染层按该节点确定，允许其并非第一渲染层。

多视图、边界裁切、跨场景或嵌套 Render Layers、Cryptomatte、序列编辑器输出、动态影像依赖、活动模拟缓存、自定义 OCIO 及可执行内嵌脚本尚不支持，会明确报错。EEVEE/Cycles 内部已经执行的效果仍处于渲染器内部；此处的“后处理之前”指原合成器入口，不是渲染器内部全部效果之前。

| 文件 | 内容 |
|---|---|
| `scene_snapshot.blend` | 本次场景快照 |
| `engine_raw.exr` | 同次渲染的 Combined 和原图所需通道 |
| `native_reference.png`, `native_linear.exr` | 同次原生后处理参考 |
| `original_rebuilt.png`, `identity.png`, `neural.png` | 原色、身份、神经分支最终图像；Neural 按模式生成 |
| `identity_hdr.exr`, `neural_hdr.exr` | 原合成器之前的分支 HDR |
| `capture_manifest.json`, `pipeline_report.json` | 捕获绑定和验收报告 |
| `worker.log`, 阶段 `.log`, `error.json` | 执行记录和错误；错误文件仅在失败时生成 |

该模型是公开恢复的静态实现；插件不提供原生动态 DLSS ABI、实时 GPU 纹理共享或动画渲染调度。模型可能增加颗粒，管线一致性通过不代表输出画质提高。

### 源码与打包

此仓库直接包含插件源码。先创建 `dist` 目录，再通过 Blender 官方命令构建并校验安装包：

```bash
blender --command extension build --source-dir . --output-dir dist
blender --command extension validate dist/neural_render_bridge-0.1.0.zip
```

外部 Python 的依赖列表位于 `requirements.txt`。安装包与 SHA-256 校验文件发布在本仓库的 [Releases](https://github.com/hpoc766-afk/neural-render-bridge/releases) 中。许可证为 GPL-3.0-or-later，详见 `LICENSE`。

## English

### Purpose and insertion point

Neural Render Bridge is a single-frame neural rendering extension for Blender 5.1. It reads same-frame scene-linear HDR **after native rendering and before the original compositor**, processes color with PyTorch, then returns HDR RGB to the original node graph. Original AOVs, masks, alpha, compositor nodes, and display transforms remain part of the final output.

The network consumes the Render Layers Combined buffer. Final PNG files are used for viewing results, never as inference inputs.

```text
Scene snapshot → One native render → Same-frame Combined HDR + consumed passes
                                       ├─ Original color ─────→ Original compositor → Original
                                       ├─ Identity color ─────→ Original compositor → Identity
                                       └─ PyTorch → HDR RGB ──→ Original compositor → Neural
```

### Installation and dependencies

1. Open **Edit → Preferences → Get Extensions** in Blender. Choose **Install from Disk** in the menu, install `neural_render_bridge-0.1.0.zip`, and enable the extension.
2. Prepare external Python 3.10 or newer with a CUDA-enabled PyTorch build suitable for your GPU and driver, plus `numpy`, `Pillow`, and `OpenEXR`. Install these in external Python, not Blender's embedded interpreter.
3. Provide model source and a compatible `dlss5_static.pt` checkpoint. The model source directory must directly contain `dlss5/graph.py`, such as the model repository's `src` directory. The upstream implementation is [taowen/dlss5-onnx](https://github.com/taowen/dlss5-onnx).
4. Configure **Python executable**, **Model source directory**, **Checkpoint**, and optionally **Output directory** in the extension preferences.
5. Click **Check Environment**. This checks module imports and, in neural mode, CUDA availability and checkpoint existence. Checkpoint structure is validated when the model is actually loaded.

The installation archive includes the Blender UI, job manager, and rendering bridge. PyTorch, model source, weights, and engine binaries are supplied locally and are excluded from the archive. Identity mode and strength-zero mode do not load weights, but still require importable Python dependencies and model source.

Install the non-PyTorch dependencies in the external environment using the following command. Select the CUDA PyTorch build using the [official instructions](https://pytorch.org/get-started/locally/).

```bash
python -m pip install numpy Pillow OpenEXR
```

### Usage

Open **Properties → Render → Neural Render Bridge**, select a mode, and click **Render with Neural Pipeline**.

| Setting | Behavior |
|---|---|
| Neural | Run PyTorch inference and replay the original compositor |
| Identity validation | Preserve HDR and validate capture/compositor replay |
| Strength | Scale the HDR residual from 0 to 1; zero skips inference |
| Eager inference | Use ordinary PyTorch execution instead of CUDA Graph |

The extension uses the current scene, current integer frame, and the view layer selected by the original compositor's Render Layers node. It writes an independent `.blend` snapshot of in-memory scene data and dependencies, remapping external paths to absolute paths without overwriting the open file. Save or pack file textures with unsaved pixel edits before running.

Execution happens in the background, with the current stage displayed in the panel. **Cancel Job** terminates only the worker and child processes launched for that job. Use **View Original / View Identity / View Neural** to load completed images, or **Open Output Folder** to inspect all records.

Use the extension's render button. Ordinary F12 retains Blender's native behavior; the extension does not register a replacement scene RenderEngine or change the F12 shortcut. Results are stored in the job directory and can be loaded into the Image Editor.

### Implementation

**Process separation.** The interactive Blender process handles preferences, scene snapshots, job status, and image viewing. An external Python worker loads the model and invokes background instances of the same Blender executable for rendering and compositor replay. PyTorch never runs inside the interactive Blender process.

**Capture boundary.** The bridge adds File Output capture nodes to the original Render Layers outputs in the background scene. Combined and the auxiliary passes consumed by the original graph are written as lossless ZIP-compressed, FLOAT32 multilayer OpenEXR. All captured data belongs to one source render. The final compositor input is also exported during that render as an independent native reference. No previously saved PNG is fed to the compositor as the source of neural inference.

**Model adaptation.** The adapter forms a nonnegative model input from native HDR RGB, applies exposure and Reinhard mapping, then resizes proportionally and pads by replicating edges to 256×256. The public static network uses NCHW, FP16, and optional CUDA Graph execution. Its output is cropped and resized to the native resolution. The static frontend substitutes current color for history and keeps the public defaults for unknown scalar and auxiliary features; depth and motion vectors are not used to reconstruct the native temporal interface.

**HDR restoration.** A bounded inverse Reinhard mapping is applied to model input and output. Their difference becomes a residual anchored to the original HDR:

```text
restored_HDR = original_HDR + strength × (inverse(output) − inverse(input))
```

The inverse is limited to the adapter's HDR range; the original HDR base is not clipped. A zero residual therefore retains negative and bright native values. Alpha is copied from the source capture, and auxiliary passes bypass the network.

**Original postprocessing.** Replay replaces the Render Layers node with an Image node reading the same-frame multilayer EXR. Only RGB connections originally coming from its `Image` socket are redirected to the neural HDR image. Depth, normals, AOVs, and masks retain cached source data. A dedicated replay RenderEngine forbids geometry rendering and runs only the original compositor. Original color management provides the final display encoding.

**Validation and cache binding.** Capture metadata binds the scene snapshot, frame, camera, view layer, dimensions, settings, external dependencies, code hashes, and resource hashes. Branch inputs additionally bind the capture ID and source alpha hash. Both passthrough and identity branches must match the same-render native reference in floating-point and PNG comparisons. The maximum budgets are one 8-bit PNG code value and `1e-6` in floating-point. Unsupported structures and failed comparisons stop execution and retain errors and stage logs.

### Supported scope and outputs

The verified environment is Windows, Blender 5.1.2, CUDA PyTorch, and an NVIDIA GPU. Blender 5.1 is required; neural inference requires a CUDA-compatible GPU. Current support covers single frames, full-frame rendering, Linear Rec.709, an enabled original compositor, and exactly one Render Layers source owned by the current scene. Its selected view layer may be any enabled layer, including a non-first layer.

Multiview, border/cropped rendering, cross-scene or nested Render Layers sources, Cryptomatte, sequencer output, animated image dependencies, active simulation caches, custom OCIO, and executable embedded scripts are unsupported and produce explicit errors. Effects already performed internally by EEVEE/Cycles remain inside the renderer. “Before postprocessing” here means before the original compositor, rather than before every internal renderer effect.

| File | Contents |
|---|---|
| `scene_snapshot.blend` | Scene snapshot for this job |
| `engine_raw.exr` | Same-render Combined and required auxiliary passes |
| `native_reference.png`, `native_linear.exr` | Same-render native postprocessing reference |
| `original_rebuilt.png`, `identity.png`, `neural.png` | Final original, identity, and neural branches; Neural is mode-dependent |
| `identity_hdr.exr`, `neural_hdr.exr` | Branch HDR before the original compositor |
| `capture_manifest.json`, `pipeline_report.json` | Capture identity and validation results |
| `worker.log`, stage `.log` files, `error.json` | Execution and failure records; error.json is created only on failure |

The model is a publicly recovered static implementation. This extension does not provide the native dynamic DLSS ABI, real-time GPU texture sharing, or animation scheduling. The model can introduce grain; successful pipeline validation does not establish improved image quality.

### Source and packaging

This repository contains the extension source directly. Create the `dist` directory, then build and validate with Blender's official commands:

```bash
blender --command extension build --source-dir . --output-dir dist
blender --command extension validate dist/neural_render_bridge-0.1.0.zip
```

External Python dependencies are listed in `requirements.txt`. Installation archives and SHA-256 checksums are published in this repository's [Releases](https://github.com/hpoc766-afk/neural-render-bridge/releases). The license is GPL-3.0-or-later; see `LICENSE`.
