# Neural Render Bridge

[中文](#中文) · [English](#english)

[GitHub](https://github.com/hpoc766-afk/neural-render-bridge) · [Download / 下载](https://github.com/hpoc766-afk/neural-render-bridge/releases/latest)

## 中文

### 功能与接入位置

Neural Render Bridge 是 Blender 5.1 的单帧神经渲染插件。它在**渲染器输出之后、原合成器后处理之前**读取同帧 Combined HDR，用 PyTorch 处理颜色，将恢复后的 HDR RGB 交回原合成图。AOV、遮罩、alpha 和显示变换沿用原场景。

合成器中的 **Neural Render Bridge** 节点提供模型文件夹、权重选择和运行按钮。用户可以指定本地目录，从下拉列表选择其中兼容的 `.pt`、`.pth` 或 `.ckpt` 权重。节点保存所选文件名，后台加载这个文件；节点配置优先于偏好设置和 Render 面板中的默认配置。

```text
Render Layers / Image → Neural Render Bridge → 原合成器后处理 → 最终图像
                              ↑
                      模型文件夹 / 权重选择
```

网络输入来自 Render Layers 的 Combined 缓冲，最终 PNG 用于查看结果。网络分支只替换进入桥接节点的 RGB；直接从 Render Layers 接到其他节点的旁路仍使用原色。

### 安装与依赖

1. 在 **Edit → Preferences → Get Extensions** 中选择 **Install from Disk**，安装 `neural_render_bridge-0.2.0.zip` 并启用插件。
2. 准备独立的 Python 3.10 或更高版本，安装匹配显卡与驱动的 CUDA PyTorch，以及 `numpy`、`Pillow`、`OpenEXR`。这些依赖安装在外部 Python 中。
3. 准备兼容的模型源码和权重。模型源码目录必须直接包含 `dlss5/graph.py`，例如 [taowen/dlss5-onnx](https://github.com/taowen/dlss5-onnx) 的 `src` 目录。把兼容此静态网络架构的 PyTorch 权重放到一个本地文件夹。
4. 在插件偏好设置中填写 **Python executable**，可选填写 **Output directory**。偏好设置中的 **Model source directory** 和 **Checkpoint** 用于无桥接节点时的运行，也作为新建节点的初始值。

安装包包含 Blender UI、任务管理器与渲染桥接源码。PyTorch、模型源码、权重和引擎二进制由本机提供。扩展名符合筛选条件不代表权重结构兼容；实际加载时会验证结构。目前模型适配器支持公开恢复的 DLSS5 静态网络架构。

非 PyTorch 依赖可通过以下命令安装；CUDA PyTorch 请按 [官方安装说明](https://pytorch.org/get-started/locally/) 选择版本。外部环境的依赖列表也位于 `requirements.txt`。

```bash
python -m pip install numpy Pillow OpenEXR
```

### 节点使用

1. 启用当前场景的合成器，保留已连接的 Render Layers 和原后处理。
2. 在合成器的 **Add → Neural Render Bridge** 中添加节点，或在 **Properties → Render → Neural Render Bridge** 中点击 **Add Model Selection Node**。添加操作将节点插入现有 Render Layers / Image 连接之间，辅助通道的连接不变。
3. 在节点的 **Model folder** 中选择权重文件夹，在 **Model** 下拉列表中选择权重，在 **Model source** 中选择模型源码目录。列表显示所选文件夹直接包含的 `.pt`、`.pth` 和 `.ckpt` 文件，不递归扫描子目录。路径可使用 Blender 的 `//` 相对路径。
4. 点击节点中的 **Check Environment** 检查模块导入；神经模式也检查 CUDA 和所选文件是否存在。
5. 选择模式并点击节点的 **Render Neural Pipeline**。也可使用 Render 面板中的运行按钮，它会读取同一个节点配置。

| 节点设置 | 含义 |
|---|---|
| Model folder | 用户指定的本地权重文件夹 |
| Model | 文件夹中的权重文件，选择结果保存在 `.blend` 中 |
| Model source | 包含 `dlss5/graph.py` 的外部模型源码目录 |
| Neural | 加载节点所选权重，执行 PyTorch 推理 |
| Identity validation | 保持 HDR 不变，验证捕获与原合成器回放 |
| Strength | HDR 残差强度，范围 0–1；0 跳过模型推理 |
| Eager inference | 使用普通 PyTorch 前向代替 CUDA Graph |

节点必须位于顶层合成图中，输入直接连接 **Render Layers / Image**，输出连接原后处理；每个合成图支持一个桥接节点。断开、静音或修改节点内部直通组会明确报错。已选模型缺失时任务停止，不会改用偏好设置中的其他权重。

运行使用当前场景、当前整数帧和 Render Layers 指定的渲染层。当前内存场景及依赖写入独立 `.blend` 快照，外部路径转为绝对路径；原文件不会被覆盖，主窗口中的路径不变。文件纹理有未保存的像素编辑时，需要先保存或打包纹理。

任务在后台执行，Render 面板显示阶段。**Cancel Job** 终止该任务启动的进程树。完成后通过 **View Original / View Identity / View Neural** 加载图像，或通过 **Open Output Folder** 查看记录。

**使用节点或面板的插件运行按钮启动神经链路。** 节点内部是原色直通组，普通 F12 仍执行原生渲染与直通合成；插件没有替换 F12 或场景 RenderEngine。真实网络由插件按钮启动的独立工作进程执行，结果保存到任务目录并可加载到 Image Editor。

### 实现方式

**节点前端。** 节点继承 `CompositorNodeCustomGroup`，保存权重目录、选中文件名、模型源码路径和推理设置。动态枚举扫描用户选择的目录，所选文件名独立持久化。运行入口解析节点路径，将所选权重的绝对路径和节点名传给工作进程。模型源码和权重不会复制进安装包。

**进程分工。** 主 Blender 负责节点 UI、场景快照、任务状态和结果加载。外部 Python 加载 PyTorch 与模型，调用同一 Blender 可执行文件的后台实例。后台在打开快照前注册桥接节点类型，保证自定义节点可以正确恢复；主 Blender 不导入 PyTorch。

**捕获边界。** 在原 Render Layers 输出上增加 File Output 捕获节点，保存同次源渲染的 Combined 和原合成图消费的辅助通道，格式为 FLOAT32、无损 ZIP 压缩多层 OpenEXR。同次渲染也导出原生后处理参考。源渲染只执行一次，缓存绑定场景、帧、相机、层、设置、依赖、代码与资源摘要。

**模型适配。** 从原始 HDR RGB 建立非负模型输入，执行曝光与 Reinhard 映射，等比例缩放并复制边缘补齐到 256×256，使用 NCHW、FP16 和可选 CUDA Graph。输出去掉补边并恢复原尺寸。公开静态前端用当前颜色替代历史，未知标量与辅助特征保持默认值；深度和运动通道保留给原合成器，不用于恢复原生时序 ABI。

**HDR 回写。** 对模型输入和输出做有界逆 Reinhard，以二者之差作为锚定原始 HDR 的残差：

```text
restored_HDR = original_HDR + strength × (inverse(output) − inverse(input))
```

原始 HDR 基底不裁切，零残差保留负值与高亮值。alpha 保持原捕获值，辅助通道不经过网络。

**后处理回接。** 回放把 Render Layers 替换为同帧多层 EXR Image，并仅将桥接节点的 Image 入口改接网络 HDR。其他 RGB 旁路及辅助通道使用源缓存。节点内部直通组把该 HDR 传给原后处理，原颜色管理负责最终显示编码。专用回放 RenderEngine 禁止几何渲染，避免第二次渲染源场景。

**验收。** 原色与身份分支必须匹配同次原生参考，浮点最大差阈值为 `1e-6`，PNG 阈值为一个 8-bit 编码值。分支输入绑定捕获 ID 与原 alpha 摘要，报告记录实际加载的权重路径、SHA-256 和被替换的连接。验证失败或不支持的结构会停止任务并保留日志。

### 支持范围与输出

已验证环境为 Windows、Blender 5.1.2、CUDA PyTorch 与 NVIDIA GPU。当前支持单帧、完整画幅、Linear Rec.709 和恰好一个本场景 Render Layers 源；该源可指向非第一渲染层。身份与强度为零的模式跳过权重加载，但仍需要可导入的 Python 依赖与模型源码。

多视图、边界裁切、跨场景或嵌套 Render Layers、嵌套桥接节点、Cryptomatte、序列编辑器输出、动态影像依赖、活动模拟缓存、自定义 OCIO 和内嵌可执行脚本不受支持。EEVEE/Cycles 内部效果仍在渲染器内部；接入边界是原合成器之前。

| 文件 | 内容 |
|---|---|
| `scene_snapshot.blend` | 当前内存场景快照与节点配置 |
| `engine_raw.exr` | 同次 Combined 与辅助通道 |
| `native_reference.png`, `native_linear.exr` | 同次原生后处理参考 |
| `original_rebuilt.png`, `identity.png`, `neural.png` | 原色、身份与神经最终结果；Neural 按模式生成 |
| `identity_hdr.exr`, `neural_hdr.exr` | 后处理之前的分支 HDR |
| `capture_manifest.json`, `pipeline_report.json` | 捕获绑定、模型路径与验收报告 |
| `worker.log`, 阶段 `.log`, `error.json` | 执行记录，错误文件仅在失败时生成 |

插件不提供原生动态 DLSS ABI、实时 GPU 纹理共享或动画调度。模型可能增加颗粒，一致性通过不代表画质提高。

### 源码与打包

先创建 `dist` 目录，使用 Blender 官方命令构建和校验：

```bash
blender --command extension build --source-dir . --output-dir dist
blender --command extension validate dist/neural_render_bridge-0.2.0.zip
```

安装包与 SHA-256 发布在本仓库的 [Releases](https://github.com/hpoc766-afk/neural-render-bridge/releases) 中。许可证为 GPL-3.0-or-later，见 `LICENSE`。

## English

### Purpose and insertion point

Neural Render Bridge is a single-frame rendering extension for Blender 5.1. It reads same-frame Combined HDR **after native rendering and before the original compositor**, processes color with PyTorch, and returns restored HDR RGB to the original graph. Source AOVs, masks, alpha, and display transforms are retained.

The **Neural Render Bridge** compositor node provides a model folder, checkpoint selector, and render button. Choose a local directory and select a compatible `.pt`, `.pth`, or `.ckpt` checkpoint from its dropdown. The selected filename is persisted, and the worker loads that file. Node settings take precedence over preference and Render panel defaults.

```text
Render Layers / Image → Neural Render Bridge → Original postprocessing → Final image
                                ↑
                     Model folder / Checkpoint selector
```

Inference consumes the Render Layers Combined buffer; final PNGs are for viewing. Only RGB entering the bridge node is replaced. Other connections directly from Render Layers retain the original color.

### Installation and dependencies

1. In **Edit → Preferences → Get Extensions**, choose **Install from Disk**, install `neural_render_bridge-0.2.0.zip`, and enable it.
2. Prepare external Python 3.10 or newer with a CUDA PyTorch build suitable for the GPU and driver, plus `numpy`, `Pillow`, and `OpenEXR`. Install these in external Python.
3. Supply compatible model code and weights. The source directory must directly contain `dlss5/graph.py`, such as `src` in [taowen/dlss5-onnx](https://github.com/taowen/dlss5-onnx). Put checkpoints compatible with this static network architecture in a local folder.
4. Set **Python executable** and optionally **Output directory** in extension preferences. Preference **Model source directory** and **Checkpoint** fields support operation without a bridge node and initialize newly added nodes.

The archive includes Blender UI, job management, and bridge source. PyTorch, model code, weights, and engine binaries are supplied locally. A matching file extension does not establish checkpoint compatibility; structure is checked during loading. The current adapter supports the publicly reconstructed static DLSS5 architecture.

Install the non-PyTorch packages using the command below. Select CUDA PyTorch using the [official instructions](https://pytorch.org/get-started/locally/). External dependencies are also listed in `requirements.txt`.

```bash
python -m pip install numpy Pillow OpenEXR
```

### Node usage

1. Enable the current scene's compositor, preserving connected Render Layers and postprocessing nodes.
2. Choose **Add → Neural Render Bridge** in the compositor, or **Add Model Selection Node** in **Properties → Render → Neural Render Bridge**. The operator inserts the bridge into existing Render Layers / Image connections. Auxiliary connections are retained.
3. Select the checkpoint directory in **Model folder**, choose a checkpoint in **Model**, and provide its code directory in **Model source**. The dropdown lists `.pt`, `.pth`, and `.ckpt` files directly in the selected folder, without recursing into subfolders. Blender `//` relative paths are supported.
4. Use the node's **Check Environment** button to check imports; neural mode also checks CUDA and selected file existence.
5. Choose a mode and click **Render Neural Pipeline** in the node. The Render panel button reads the same node configuration.

| Node setting | Behavior |
|---|---|
| Model folder | User-selected local checkpoint directory |
| Model | Checkpoint in that folder, persisted in the `.blend` file |
| Model source | External code directory containing `dlss5/graph.py` |
| Neural | Load the selected checkpoint and run PyTorch |
| Identity validation | Preserve HDR and validate capture/compositor replay |
| Strength | HDR residual multiplier from 0–1; zero skips inference |
| Eager inference | Use ordinary PyTorch instead of CUDA Graph |

The node must be in the top-level compositor, directly connected to **Render Layers / Image**, with its output connected to the original postprocessing. One bridge node is supported per graph. Disconnected or muted nodes and modified internal passthrough groups are rejected. Missing selected models stop execution without falling back to a preference checkpoint.

Execution uses the current scene, integer frame, and the view layer selected by Render Layers. An independent `.blend` snapshot captures in-memory scene data and dependencies with external paths remapped to absolute paths. The open file is not overwritten and interactive paths remain unchanged. Save or pack file textures containing unsaved pixel edits first.

Jobs run in the background, with their stage shown in the Render panel. **Cancel Job** terminates the job's own process tree. Use **View Original / View Identity / View Neural** to load results or **Open Output Folder** to inspect records.

**Start neural processing with the extension's node or panel button.** The node contains a native-color passthrough group. Ordinary F12 performs native rendering and passthrough compositing; no F12 shortcut or scene RenderEngine is replaced. Actual inference runs in the external worker launched by the extension button. Results are saved to the job directory and can be loaded into the Image Editor.

### Implementation

**Node frontend.** A `CompositorNodeCustomGroup` stores the model folder, selected filename, source path, and inference settings. A dynamic enum scans the selected directory; the filename is independently persisted. The launch entry resolves node paths and passes the selected checkpoint's absolute path and node name to the worker. Model code and weights are not copied into the installation archive.

**Processes.** Interactive Blender handles node UI, snapshots, job status, and result loading. External Python imports PyTorch and invokes background instances of the same Blender executable. Background stages register the bridge node type before loading the snapshot, restoring custom nodes correctly. Interactive Blender never imports PyTorch.

**Capture.** File Output nodes attached to the original Render Layers outputs capture Combined and consumed auxiliary passes as lossless ZIP-compressed FLOAT32 multilayer OpenEXR. All data and the native postprocessed reference come from one source render. Capture metadata binds the scene, frame, camera, layer, settings, dependencies, code, and resource hashes.

**Model adaptation.** Native HDR RGB forms a nonnegative model input with exposure and Reinhard mapping. Proportional resizing and replicated edge padding produce 256×256 NCHW input for FP16 inference with optional CUDA Graph. Output is cropped and resized to native dimensions. The public static frontend substitutes current color for history and keeps unknown feature defaults. Source depth and motion remain available to the compositor; they do not restore the native temporal ABI.

**HDR restoration.** A bounded inverse Reinhard mapping transforms input and output, whose difference becomes a residual anchored to the native HDR:

```text
restored_HDR = original_HDR + strength × (inverse(output) − inverse(input))
```

The original HDR base is not clipped. Zero residual preserves negative and bright values. Alpha retains source values; auxiliary passes bypass inference.

**Postprocessing.** Replay replaces Render Layers with a same-frame multilayer EXR Image node and redirects only the bridge node's Image input to neural HDR. RGB bypasses and auxiliary routes retain source cache data. The bridge's internal passthrough group forwards the injected HDR to original postprocessing, and original color management provides display encoding. A dedicated replay RenderEngine prohibits geometry rendering.

**Validation.** Original and identity branches must match the native reference within `1e-6` floating-point difference and one 8-bit PNG code value. Branches bind the capture ID and alpha hash. Reports record the loaded checkpoint path, SHA-256, and replaced connections. Failed checks or unsupported structures stop the job and retain logs.

### Scope and outputs

Validated with Windows, Blender 5.1.2, CUDA PyTorch, and an NVIDIA GPU. Supports a single frame, full frame extent, Linear Rec.709, and exactly one local Render Layers source, including a source using a non-first view layer. Identity and strength-zero modes skip checkpoint loading but still require importable dependencies and model code.

Unsupported cases include multiview, border crops, cross-scene or nested Render Layers, nested bridge nodes, Cryptomatte, sequencer output, dynamic image dependencies, active simulation caches, custom OCIO, and executable embedded scripts. EEVEE/Cycles internal effects remain inside the renderer; the insertion boundary is before the original compositor.

| File | Contents |
|---|---|
| `scene_snapshot.blend` | In-memory scene snapshot and node settings |
| `engine_raw.exr` | Same-render Combined and auxiliary passes |
| `native_reference.png`, `native_linear.exr` | Same-render native reference |
| `original_rebuilt.png`, `identity.png`, `neural.png` | Original, identity, and optional neural results |
| `identity_hdr.exr`, `neural_hdr.exr` | Branch HDR before postprocessing |
| `capture_manifest.json`, `pipeline_report.json` | Capture bindings, selected model path, and validation |
| `worker.log`, stage `.log`, `error.json` | Execution logs; error JSON exists only on failure |

Native dynamic DLSS ABI, real-time GPU texture sharing, and animation scheduling are not provided. The model can introduce grain; pipeline consistency does not establish improved image quality.

### Source and packaging

Create `dist`, then use Blender's official build and validation commands:

```bash
blender --command extension build --source-dir . --output-dir dist
blender --command extension validate dist/neural_render_bridge-0.2.0.zip
```

Installation archives and SHA-256 checksums are published in this repository's [Releases](https://github.com/hpoc766-afk/neural-render-bridge/releases). Licensed under GPL-3.0-or-later; see `LICENSE`.
