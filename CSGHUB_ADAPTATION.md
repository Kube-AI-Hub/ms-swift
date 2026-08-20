# ms-swift 适配 CSGHub 移植指南

本文档说明 `origin/release/4.1` 分支（`Kube-AI-Hub/ms-swift`）在上游 `modelscope/ms-swift` v4.1.0（tag `v4.1.0`，commit `5b40cf2b43051f4`）基础上为适配 CSGHub 所做的全部修改，用于指导在**新版本 ms-swift**（如上游 `main` / 4.5.x）上重放这些适配。

> **状态**：`release/4.5` 分支已按本指南从上游 `v4.5.2` 完成移植（四组改动全部落地，`app.py` 默认值字典已核对与 v4.5.2 一致、无需修改）。文中「4.5.x 注意/风险点」小节记录了移植时实际遇到的差异与取舍。镜像 Dockerfile 尚未切换到该分支，切换时更新其中 pin 的 commit。

改动后的代码用于构建 csghub-server 中的三个微调镜像（提供 Web UI RESTful + Jupyter Notebook 的 ms-swift 微调环境）：

- `csghub-server/docker/finetune/Dockerfile.ms-swift`（NVIDIA CUDA）
- `csghub-server/docker/finetune/Dockerfile.ms-swift-npu`（华为昇腾 NPU）
- `csghub-server/docker/finetune/Dockerfile.ms-swift-mlu`（寒武纪 MLU）

---

## 1. 改动总览

从 `v4.1.0` 到 `release/4.1` HEAD（`f103023c6`）共 15 个 commit，38 个文件，约 +9561/-122 行。按功能分为四组，**移植时建议按组处理**：

| 组 | 内容 | 侵入性 | 文件数 |
|---|---|---|---|
| A | `web_ui_restful` 新模块（FastAPI + 静态前端） | 低（几乎纯新增） | 12 |
| B | 多硬件支持（寒武纪 MLU + 单卡 UI 设备标签） | 中（改公共工具 + 7 个 Gradio UI 文件） | 10 |
| C | CSGHub Hub 集成（读级联 / 写路由） | **高（核心侵入式改动）** | 14 |
| D | 训练稳定性补丁（Template pickle） | 低（独立小改动） | 1 |

各 commit 与文件的对应关系（按时间顺序）：

| commit | 说明 | 组 |
|---|---|---|
| `2f3a249dc` | feat: add RESTful web UI with TensorBoard and Gradio integration | A+B+D |
| `48698809a` | feat: hub download priority（tuners 改用 safe_snapshot_download） | C |
| `d59a8f50e` | fix(web-ui-restful): derive TensorBoard open_url from request base URL | A |
| `86f62d079` | fix(hub): support pycsghub snapshot_download import name | C |
| `bd295b075` | feat: add hub download fallback priority | C |
| `0266b0790` | feat(hub): cascade reads and route writes to CSGHub（**核心**） | C |
| `6139503c6` | fix(hub): normalize CSGHub revision for pycsghub snapshot_download | C |
| `11f095d27` | fix(args): keep dataset reads on cascade when USE_HF=1 is set | C |
| `539e9facc` | fix(web_ui_restful): historical records show logs, metrics, and TB URLs | A |
| `a131cea29` | fix(hub): only forward HF_TOKEN to CSGHub in read cascades | C |
| `b3aac5e27` | fix(web_ui_restful): default train dataset to self-cognition（仅前端） | A |
| `b55bd210d` | fix(web_ui_restful): avoid reusing old train log dirs（仅前端） | A |
| `01724cbdc` | fix(web_ui_restful): show gpu/npu labels like mps for single-device UI | A |
| `be8057dac` | fix(web_ui_restful): avoid UnboundLocalError in _build_gpu_env on NPU | A |
| `f103023c6` | feat: add Cambricon MLU support to torch_utils | B |

> 注意：三个 Dockerfile 目前 pin 的是 `be8057dac`（倒数第二个 commit），**不包含** MLU torch_utils 补丁 `f103023c6`。移植完成后更新 Dockerfile 中的 commit 时请 pin 新分支 HEAD。

---

## 2. 组 A：`web_ui_restful` 新模块

### 2.1 新增文件（直接整体拷贝，无冲突风险）

```
swift/ui/web_ui_restful/__init__.py          # from .app import webui_restful_main
swift/ui/web_ui_restful/app.py               # FastAPI 应用（~1600 行，所有 /api/v1/* 端点）
swift/ui/web_ui_restful/utils.py             # psutil 进程发现 / kill / 日志 tail
swift/ui/web_ui_restful/tensorboard_manager.py # TensorBoard 子进程管理 + 反代辅助
swift/ui/web_ui_restful/tb_series.py         # TensorBoard scalar 序列（与 Gradio Runtime.*_plot 对齐）
swift/ui/web_ui_restful/frontend/index.html  # 静态前端（单页，~2200 行）
swift/ui/web_ui_restful/frontend/app.js      # 前端逻辑（~2900 行）
swift/ui/web_ui_restful/frontend/index.css
swift/ui/web_ui_restful/frontend/index-i18n.js
swift/cli/web_ui_restful.py                  # CLI 入口
```

前端是自包含的（无构建步骤、无 npm 依赖），`apiBase` 从 `window.location.pathname` 推导，因此在 Jupyter `/proxy/7860/` 这类子路径反代下无需改动即可工作。

### 2.2 需要在既有文件中重放的小改动

1. **`swift/cli/main.py`** — `ROUTE_MAPPING` 增加一行：

```python
'web-ui-restful': 'swift.cli.web_ui_restful',
```

2. **`setup.py`** — `package_data` 增加前端静态文件：

```python
package_data={'': ['utils/*', 'dataset/data/*.*', 'config/*.json', 'loss_scale/config/*.json',
                   'ui/web_ui_restful/frontend/*']},
```

3. **`swift/arguments/webui_args.py`** — `WebUIArguments` 增加两个字段：

```python
thread_pool_workers: Optional[int] = None      # asyncio 默认线程池上限
tensorboard_path_prefix: str = '/tensorboard'  # TB 反代路径前缀
```

### 2.3 模块对 swift 内部 API 的依赖（移植时逐项核对）

`app.py` / `utils.py` / `tb_series.py` 依赖以下内部符号，新版本上要确认仍然存在或调整 import：

| 依赖 | 用途 | 4.5.x 现状 |
|---|---|---|
| `swift.utils`: `get_logger`, `get_device_count`, `format_time`, `read_tensorboard_file`, `tensorboard_smoothing` | 通用工具 | 仍存在 |
| `swift.utils`: `get_ui_device_info`, `is_torch_mlu_available` | **由组 B 新增**，须先移植组 B | 需自行添加 |
| `swift.arguments.WebUIArguments` | 启动参数 | 仍存在（需加 2.2-3 字段） |
| `swift.ui.llm_train` 的 `run_command_in_background_with_popen` | 后台拉起 swift 子命令 | 上游已把实现移到 `swift/ui/llm_train/utils.py`；`app.py` 顶部已有 try/except 双路 import，**兼容无需修改** |
| `swift.model`: `get_model_list`, `MODEL_MAPPING`, `ModelType`, `get_matched_model_meta` | `/api/v1/models` 等 | 仍存在 |
| `swift.template.TEMPLATE_MAPPING` | 模板列表 / default_system | 仍存在 |
| `swift.dataset.get_dataset_list` | 数据集列表 | 仍存在 |

第三方依赖：`fastapi`、`uvicorn`、`pydantic`、`httpx`（TB 反代与 chat 转发必需）、`psutil`、`tensorboard`。这些在 ms-swift 依赖链里基本已带上，镜像中无需单独安装，但精简环境下要确认 `httpx` 存在。

### 2.4 关键行为约定（前后端 / 与 csghub 镜像的契约，改动时勿破坏）

- 服务端口：默认 `--server_port 7861`；环境变量 **`WEBUI_PORT`** 优先（csghub `start.sh` 设为 7860）。`WEBUI_SERVER`、`WEBUI_THREAD_POOL_WORKERS`、`WEBUI_TENSORBOARD_PREFIX` 同理。
- 训练命令拼接：与 Gradio 版对齐——参数值等于 swift 默认值时**不下发**（`SWIFT_TRAIN_DEFAULTS` 等字典）；新版本若上游默认值变化（如 `lora_rank`、`lr_scheduler_type`），需同步更新 `app.py` 里的 `SWIFT_TRAIN_DEFAULTS` / `RLHF_DEFAULTS` / `GRPO_DEFAULTS`。
- 使用 `--tuner_type`（不要用已废弃的 `--train_type`）；固定追加 `--add_version False --output_dir ... --logging_dir ... --ignore_args_error True --log_file ...`。
- 任务发现基于 `psutil` 扫描进程 cmdline 中的 `swift` + 子命令（`sft/pt/rlhf/deploy/export/eval/sample`），kill 基于 `pkill -f <log_file>`；GPU 占用从子进程 env（`CUDA/ASCEND_RT/MLU_VISIBLE_DEVICES`）读取。若新版本 CLI 拉起方式变化（如包一层 torchrun），需回归验证 `/api/v1/tasks`。
- TensorBoard：由服务端拉起 `tensorboard --path_prefix <tb_prefix>` 子进程并在同端口下做 HTTP 反代（`/tensorboard/*`、`/data/*`、`/font-roboto/*` 三组路由）。
- 训练记录持久化在 `<modelscope cache>/swift-web-ui` 目录（与 Gradio BaseUI 同一目录），按 `scope--model-timestamp` 文件存 JSON。
- csghub `start.sh` 会用 `sed` 修改 `frontend/index.html`（预填 `*-model` / `*-model-type` / `*-template` 的 value、把 `infer-port` 从 8000 改 9000）。**改前端时保持这些 input 的 id 与初始结构不变**，否则要同步更新 `csghub-server/docker/finetune/swift/start.sh`。

---

## 3. 组 B：多硬件（MLU）支持

### 3.1 `swift/utils/torch_utils.py`

1. 文件头部（`import torch` 之后立刻）增加插件式后端导入：

```python
# For Cambricon MLU: import torch first, then torch_mlu
try:
    import torch_mlu  # noqa: F401
except ImportError:
    pass
# For Huawei Ascend NPU: import torch first, then torch_npu
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
```

2. 新增两个函数：

```python
def is_torch_mlu_available() -> bool:
    try:
        import torch_mlu  # noqa: F401
    except ImportError:
        return False
    return getattr(torch, 'mlu', None) is not None and torch.mlu.is_available()

def get_ui_device_info():
    device_count = get_device_count()
    if device_count > 0:
        return [str(i) for i in range(device_count)] + ['cpu'], '0'
    elif is_torch_mps_available():
        return ['mps', 'cpu'], 'mps'
    else:
        return ['cpu'], 'cpu'
```

3. 在以下函数中增加 `elif is_torch_mlu_available():` 分支（模式与 npu 分支一致）：
   `synchronize`（`torch.mlu.synchronize`）、`get_device`（`'mlu:{}'`）、`get_current_device`、`get_torch_device`（返回 `torch.mlu`）、`set_device`、`get_device_count`（`torch.mlu.device_count()`）、`empty_cache`、`init_process_group`（backend = **`'cncl'`**）。

> 4.5.x 注意：上游 torch_utils 头部 import 已重构（`from .env import ... is_mp`，且删除了模块级 `logger`），新增了 `nanstd`/`is_torch_rocm`/`get_physical_device_count`/`ipc_collect` 等函数。MLU 分支按上述模式手工插入即可，diff 不能直接 apply。新增的 `get_physical_device_count`（NVML/ROCm 枚举）没有 MLU 路径，走 `get_device_count()` 兜底，MLU 上行为正确，可不动；如需容器内屏蔽卡场景更准确可后续补 `cndev` 枚举。

### 3.2 `swift/utils/__init__.py` 导出

**4.5.x 重大差异**：上游此文件已改为 `_LazyModule` 懒加载结构。新增导出 `get_ui_device_info`、`is_torch_mlu_available` 时必须**同时改两处**：

1. `if TYPE_CHECKING:` 块中 `from .torch_utils import (...)` 列表；
2. `else:` 分支 `_import_structure['torch_utils']` 列表。

只改一处会出现「IDE 能补全但运行时 ImportError」或反之。

### 3.3 Gradio UI 的 7 个文件（模式统一，逐个重放）

涉及：`swift/ui/llm_train/llm_train.py`、`llm_infer/llm_infer.py`、`llm_eval/llm_eval.py`、`llm_export/llm_export.py`、`llm_sample/llm_sample.py`、`llm_grpo/llm_grpo.py`、`llm_grpo/external_rollout.py`、`llm_rlhf/llm_rlhf.py`。每个文件三处同构修改：

1. import：`get_device_count` → `get_ui_device_info`；增加 `from swift.utils import is_torch_mlu_available`；
2. `do_build_ui` 中设备下拉框：

```python
# 原：
default_device = 'cpu'
device_count = get_device_count()
if device_count > 0:
    default_device = '0'
...choices=[str(i) for i in range(device_count)] + ['cpu']
# 改为：
device_choices, default_device = get_ui_device_info()
...choices=device_choices
```

3. 启动命令 env 构造处，在 `is_torch_cuda_available()` 分支后增加：

```python
elif is_torch_mlu_available():
    cuda_param = f'MLU_VISIBLE_DEVICES={gpus}'
    all_envs['MLU_VISIBLE_DEVICES'] = gpus
```

（`llm_grpo/llm_grpo.py`、`llm_rlhf/llm_rlhf.py` 继承自 `LLMTrain`，只有前两处。）

Gradio UI 在 csghub 镜像中不是主入口（主入口是 web_ui_restful），如果新版本这些文件冲突较多，**可以降级为只改 `llm_train` 与 `llm_infer`**，其余按需。

---

## 4. 组 C：CSGHub Hub 集成（核心）

### 4.1 设计约定（先理解再动手）

- **读操作**（模型下载、数据集加载）：统一走三级级联 **CSGHub(pycsghub) → ModelScope → HuggingFace(hf-mirror.com)**，逐级失败回退；`use_hf` 参数变为三态：`True` 强制 HF、`False` 强制 MS、**`None`（默认）走级联**。
- **写操作**（`push_to_hub` / `create_model_repo` / `try_login`）：**无条件走 CSGHub**（`get_write_hub()`），不受 `USE_HF` 影响。
- **`USE_HF` 环境变量不再自动提升 `args.use_hf`**：镜像里为了让部分 MLLM 辅助文件下载走 HF 而设置 `USE_HF=1`，但不允许它破坏读级联（commit `11f095d27` 修的就是这个）。
- **Token 语义**：本部署中 `HF_TOKEN` 装的是 **CSGHub token**。因此回退到 MSHub/HFHub 前必须用 `_hide_hf_token_env()` 临时摘掉 `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`/`HUGGINGFACE_HUB_TOKEN`，避免 CSGHub token 泄露给 modelscope/hf-mirror（commit `a131cea29`）。
- **Endpoint 解析**：`CSGHub._resolve_endpoint()` 优先取 `HF_ENDPOINT`（排除 huggingface.co / hf-mirror.com 两个公共值），否则用默认 `https://hub.opencsg.com/hf`。Token 顺序：`CSGHUB_TOKEN` → `HF_TOKEN` → `ACCESS_TOKEN`。
- **revision 归一化**：pycsghub 对 `revision=None` 会抛 `quote_from_bytes` 错误，`None`/`'master'` 一律归一为 `'main'`。

### 4.2 `swift/hub/hub.py`（新增 ~420 行）

新增内容全部是追加式代码，直接搬运后核对以下锚点：

1. **`class CSGHub(HubOperation)`**：实现 `try_login` / `download_model` / `load_dataset` / `create_model_repo` / `push_to_hub` / `patch_hub`。
   - 依赖 `pycsghub`（镜像里通过 `pip install csghub-sdk==0.7.6` 提供）：`pycsghub.snapshot_download.snapshot_download`、`pycsghub.repository.Repository`。
   - `patch_hub` 会临时替换 `huggingface_hub.create_repo/upload_folder` 与 `transformers.trainer.create_repo/upload_folder`。
   - **4.5.x 必须同步的修正**：上游 `MSHub.patch_hub` 已把 `trainer.create_repo` 直取改为 `getattr(trainer, 'create_repo', None)`（新版 transformers 中该属性可能不存在）。移植 `CSGHub.patch_hub` 时照抄 getattr 写法，并在 finally 恢复时判空。
   - `_build_upload_folder_shim` 依赖 `modelscope.utils.repo_utils.CommitInfo`，新版本 modelscope 若挪位置需调整。
   - 文件头已有 `from huggingface_hub import RepoUrl`（shim 用到），上游 main 仍在，无需处理。
2. **`_hide_hf_token_env()`** 上下文管理器 + `_HF_TOKEN_ENV_VARS` + `_HF_MIRROR_ENDPOINT = 'https://hf-mirror.com'`。
3. **`cascading_download_model(...)`**：CSGHub → MSHub（隐藏 HF_TOKEN，token=None）→ HFHub（临时设 `HF_ENDPOINT=hf-mirror.com`，隐藏 HF_TOKEN），全失败抛 `RuntimeError`。
4. **`cascading_load_dataset(csg_dataset_id, hf_dataset_id, ms_dataset_id, ...)`**：同上；CSGHub 层用 pycsghub 把数据集仓库 snapshot 下载到本地后再 `datasets.load_dataset(local_dir, ...)`。
5. **`get_write_hub()`**：恒返回 `CSGHub`。
6. **`get_hub(use_hf)`** 语义修改：

```python
def get_hub(use_hf: Optional[bool] = None):
    if use_hf is None:
        return CSGHub          # 原实现：use_hf = use_hf_hub()
    return {True: HFHub, False: MSHub}[use_hf]
```

7. **`swift/hub/__init__.py`**：导出改为

```python
from .hub import (CSGHub, HFHub, MSHub, cascading_download_model, cascading_load_dataset, get_hub, get_write_hub)
```

（4.5.x 该文件仍是简单 eager import，直接改。）

### 4.3 `swift/utils/hub_utils.py` — `safe_snapshot_download`

把原来的 `hub = get_hub(use_hf); hub.download_model(...)` 替换为：

```python
_token = os.getenv('HF_TOKEN') or os.getenv('ACCESS_TOKEN', '')
_download_dir = os.getenv('HUGGINGFACE_HUB_CACHE', './download')
_effective_token = _token or hub_token

with safe_ddp_context(hash_id=model_id_or_path):
    from swift.hub.hub import cascading_download_model
    model_dir = cascading_download_model(
        model_id_or_path, revision=revision, ignore_patterns=ignore_patterns,
        token=_effective_token, cache_dir=_download_dir, **kwargs)
```

`use_hf` 形参保留但被忽略（docstring 同步说明），避免改所有调用方签名。

> **4.5.x 新风险点与实际取舍（release/4.5 已实现）**：上游 `hub_utils.py` 新增了 `patch_kernels()`，其内部调用 `safe_snapshot_download(repo_id, use_hf=False, allow_patterns=...)`。最初设想是「显式 True/False 走单后端、仅 None 走级联」，但核查发现 v4.5.2 有约 11 处调用点（`_init_adapters`、各 infer engine、megatron args 等）把 **bool 型 `args.use_hf`（默认 False）无条件透传**，若 False→强制 MSHub 会让 adapter/推理引擎下载全部绕过级联。因此 release/4.5 的最终语义是：**仅 `use_hf is True`（显式 `--use_hf true`）直连 HFHub；`None`/`False` 一律走级联**。副作用是 `patch_kernels` 的 kernel 仓库也会先试 CSGHub（404 后回退 MSHub），功能可用、仅多一次失败请求；且部署中 `USE_HF=1` 时 `patched_get_kernel` 直接走原始 HF 路径，不经过 `safe_snapshot_download`。

### 4.4 `swift/arguments/base_args/base_args.py`

1. `__post_init__` 删除 USE_HF 自动提升逻辑：

```python
# 删除：
if self.use_hf or use_hf_hub():
    self.use_hf = True
    os.environ['USE_HF'] = '1'
```

2. `self.hub = get_hub(self.use_hf)` → `self.hub = get_write_hub()`（import 同步改）。

> 4.5.x 注意：该文件上游有较多无关变更（`swift.ray` → `swift.ray_utils`、新增 `_patch_peft()`、新增 `load_dataset()` 方法），**不要整体 apply diff**，只手工重放上述两点。

### 4.5 `swift/arguments/base_args/data_args.py`

`get_dataset_kwargs()`（或对应字典构造处）：

```python
'use_hf': True if self.use_hf else None,   # 原来是 self.use_hf
```

即 CLI 未显式 `--use_hf true` 时向 `load_dataset` 传 `None`（触发级联）而不是 `False`（强制 MSHub）。

### 4.6 `swift/trainers/mixin.py`

`SwiftMixin.__init__` 中 `self.hub = get_hub()` → `self.hub = get_write_hub()`（import 同步改）。4.5.x 该行仍在（约 line 112），trainer 内 `push_to_hub` / checkpoint 上传全部经由 `self.hub`。

### 4.7 数据集加载三态化

1. **`swift/dataset/loader.py`**（4.5.x 有 drift：新增 `disable_auto_column_mapping`、`_inject_dataset_routing_tag`，手工合并）：
   - `_load_repo_dataset` 增加 `dataset_meta: Optional[DatasetMeta] = None` 形参；本地目录场景不再篡改 `use_hf=True` 而是 `local_dir=True` 并直接用 `HFHub.load_dataset` 读本地；
   - `use_hf is True/False` 保持原单后端路径；`use_hf is None` 时调用 `cascading_load_dataset`，并从 `dataset_meta` 解析每个后端各自的 id/revision（`hf_dataset_id`/`ms_dataset_id`/`hf_revision`/`ms_revision`，CSGHub 复用 hf id）；
   - 模块级 `load_dataset()` 中删除 `use_hf_default = use_hf_hub()` 的兜底，保持 `None` 透传；`DATASET_MAPPING` 命中时按三态选择 canonical dataset id（None 时优先 hf id）。
2. **`swift/dataset/dataset_syntax.py`**：`get_dataset_meta(use_hf)` 接受 `Optional[bool]`，`None` 时先查 `('hf', ds)` 再查 `('ms', ds)`；目录场景查 `('repo', ds)`。（4.5.x 此文件无上游变更，可直接 apply。）
3. **`swift/dataset/register.py`**：`get_dataset_list()` 不再按 `USE_HF` 过滤，改为合并去重两个后端的 id。（4.5.x 无上游变更。）
4. **`swift/model/register.py`**：`get_model_list()` 同样合并去重 `hf_model_id` + `ms_model_id`。（4.5.x 函数仍在，周边有 trust_remote_code 等无关变更，手工重放。）

### 4.8 tuners 下载入口统一

`swift/tuners/base.py`（2 处）、`swift/tuners/peft.py`（1 处）、`swift/tuners/utils.py`（1 处）：

```python
from modelscope import snapshot_download   # 删除
from swift.utils import safe_snapshot_download   # 新增
# 调用点 snapshot_download(...) → safe_snapshot_download(...)
```

使 adapter/tuner 配置下载也走级联。4.5.x 这三个文件有少量 drift（peft 包装类扩充），锚点仍在。

### 4.9 `.gitignore`

追加 `/download`（级联下载的默认 cache 目录 `HUGGINGFACE_HUB_CACHE` 未设置时为 `./download`）。

---

## 5. 组 D：Template pickle 补丁

`swift/template/base.py` 的 `Template` 类增加：

```python
def __getstate__(self):
    # DataLoader workers (spawn / forkserver) pickle collate_fn = partial(self.data_collator, ...).
    # Omit nn.Module refs and forward-hook state: they pull in unpicklable closures from
    # PreTrainedModel.enable_input_require_grads and are not needed in workers (collate only).
    state = self.__dict__.copy()
    state['model'] = None
    state['dummy_model'] = None
    state['_handles'] = []
    state['_deepspeed_initialize'] = None
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
```

已核实上游 main 的 `Template` 仍持有 `dummy_model` / `_handles` / `_deepspeed_initialize` 属性，补丁可直接移植；但该文件上游变更极大（约 480 行），务必手工插入而非 apply diff。移植后如上游给 `Template.__init__` 增加了新的不可 pickle 属性（新的 hook/闭包），需要在 `__getstate__` 中一并剔除——验证方法见第 7 节的 dataloader 冒烟。

---

## 6. 推荐移植流程

1. **建分支**：从新版本 tag（如 `v4.5.0`）切出 `release/4.5`。
2. **先试 cherry-pick，再手工重放**。预计冲突情况：
   - 可直接 cherry-pick / apply：组 A 新文件、`dataset_syntax.py`、`dataset/register.py`、`.gitignore`、`cli/main.py`、`webui_args.py`、tuners 三文件；
   - 必须手工重放（上游 drift 大）：`utils/__init__.py`（LazyModule 双写）、`torch_utils.py`、`hub_utils.py`、`base_args.py`、`dataset/loader.py`、`template/base.py`、`model/register.py`、7 个 Gradio UI 文件；
   - `hub/hub.py`：新增代码块整体搬运 + `patch_hub` 改 getattr 写法。
3. **建议的重放顺序**：C（hub 级联，先让 CLI 训练/导出通）→ B（torch_utils + 导出）→ A（web_ui_restful）→ D（template pickle）。
4. **对齐 web_ui_restful 的默认值字典**：对照新版本 `swift/arguments` 各 dataclass 默认值更新 `app.py` 的 `SWIFT_TRAIN_DEFAULTS` / `RLHF_DEFAULTS` / `GRPO_DEFAULTS`，并核对 `RLHF_TYPE_ARGS` 与新版支持的 rlhf_type 列表。
5. **更新镜像**：三个 Dockerfile 中 `git fetch --depth 1 origin <commit>` 改为新分支 HEAD；确认基础镜像的 torch / transformers / vllm 版本与新版 ms-swift 的 `requirements` 匹配（当前镜像 pin `transformers==5.4.*`、`csghub-sdk==0.7.6`）。

---

## 7. 验证清单

镜像内（MLU 镜像先 `source /torch/venv3/pytorch/bin/activate`，NPU 镜像先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`）：

1. **导入与 CLI**：`swift web-ui-restful --server_port 7861` 能起服务；`GET /health` 返回 200 且 `devices` 列出正确设备（MLU 机器上应为 mlu 序号，单卡 CUDA/NPU 显示 `gpu`/`npu` 标签）。
2. **冒烟脚本**：跑 `csghub-server/docker/finetune/ms-swift-web-ui-restful-smoke-test.py`（TestClient 验证 `/health` 与 `dry_run` 训练命令拼接）。
3. **读级联**：
   - `HF_ENDPOINT` 指向 CSGHub 网关 + `HF_TOKEN` 有效时，`swift sft --model <csghub仓库id> ...` 从 CSGHub 下载；
   - 用一个 CSGHub 上不存在的 MS 数据集 id 验证回退到 ModelScope；断网 CSGHub 验证日志出现 "falling back to MSHub"；
   - 确认回退请求**未携带** CSGHub token（可在代理上抓包或对 hf-mirror 打日志）。
4. **写路由**：`swift export --push_to_hub true --hub_model_id <ns>/<repo>` 推到 CSGHub；`USE_HF=1` 环境下重复一次，确认仍推 CSGHub 且数据集读取仍走级联。
5. **训练全链路**：REST 发起 self-cognition LoRA SFT（数据集留空应默认 self-cognition）→ 日志 SSE → TensorBoard 启动与 `/tensorboard/` 反代 → metrics 接口 → 历史记录恢复 → 终止任务。
6. **Template pickle**：`--dataloader_num_workers 2` + spawn 场景跑通一次训练（验证 `__getstate__` 补丁在新版本仍充分）。
7. **推理验证**：REST `infer/start`（deploy 到 9000 端口）→ `infer/chat` 对话。
8. **多硬件**：三种镜像各跑一次 1–2 step 的 `max_steps` 训练，确认 `MLU_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` 注入正确、`init_process_group` backend 选择正确（cncl/hccl/nccl）。

---

## 8. 相关环境变量总表

| 变量 | 作用 | 设置方 |
|---|---|---|
| `HF_ENDPOINT` | CSGHub HF 兼容网关（如 `https://<host>/hf`）；同时被 pycsghub 读取 | csghub-server 注入 |
| `HF_TOKEN` / `CSGHUB_TOKEN` / `ACCESS_TOKEN` | CSGHub token（读写），按此优先级解析 | csghub-server 注入 |
| `HUGGINGFACE_HUB_CACHE` | 级联下载 cache 目录（默认 `./download`） | 可选 |
| `USE_HF` | 仅影响少量 MLLM 辅助文件下载；**不再影响** hub 读写路由 | `start.sh` 设为 1 |
| `MODELSCOPE_API_TOKEN` | MSHub 回退层的认证（可选） | 可选 |
| `WEBUI_SERVER` / `WEBUI_PORT` | RESTful 服务监听地址/端口（start.sh 设 7860） | `start.sh` |
| `WEBUI_THREAD_POOL_WORKERS` | 阻塞任务线程池上限 | 可选 |
| `WEBUI_TENSORBOARD_PREFIX` | TB 反代路径前缀（默认 `/tensorboard`） | 可选 |
| `MAX_LOG_LINES` | 日志 SSE 每次推送的最大行数（默认 200） | 可选 |
| `REPO_ID` / `REVISION` / `CONTEXT_PATH` | 预填模型、锚定 revision、反代前缀 | csghub-server 注入，`start.sh` 消费 |
| `GRADIO_ROOT_PATH` | Gradio web-ui 子路径（web_ui_restful 不用） | Dockerfile/`start.sh` |
