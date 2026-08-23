# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import time

import swift
from swift.arguments import WebUIArguments
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from swift.utils import get_device_count, get_logger, get_ui_device_info, is_torch_mlu_available

from .utils import (get_all_running_tasks, get_running_tasks, get_used_ports, kill_task_by_log,
                    tail_log_file)

try:
    from swift.ui.llm_train.utils import run_command_in_background_with_popen
except ImportError:
    from swift.ui.llm_train import run_command_in_background_with_popen

logger = get_logger()

FRONTEND_DIR = Path(__file__).parent / 'frontend'

# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    model: str
    model_type: Optional[str] = None
    template: Optional[str] = None
    dataset: Optional[List[str]] = None
    train_stage: str = 'sft'
    tuner_type: Optional[str] = None
    seed: Optional[int] = None
    torch_dtype: Optional[str] = None
    use_liger_kernel: bool = False
    gpu_ids: Optional[List[str]] = None
    use_ddp: bool = False
    ddp_num: int = 1
    deepspeed: Optional[str] = None
    sequence_parallel_size: Optional[int] = None
    learning_rate: Optional[float] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    num_train_epochs: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    attn_impl: Optional[str] = None
    neftune_noise_alpha: Optional[float] = None
    output_dir: Optional[str] = None
    logging_dir: Optional[str] = None
    system: Optional[str] = None
    envs: Optional[str] = None
    dry_run: bool = False
    more_params: Optional[str] = None
    # Advanced settings
    tuner_backend: Optional[str] = None
    weight_decay: Optional[float] = None
    logging_steps: Optional[int] = None
    lr_scheduler_type: Optional[str] = None
    warmup_ratio: Optional[float] = None
    truncation_strategy: Optional[str] = None
    max_steps: Optional[int] = None
    max_grad_norm: Optional[float] = None
    # LoRA params
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_dtype: Optional[str] = None
    use_rslora: bool = False
    use_dora: bool = False
    target_modules: Optional[str] = None
    # Dataset settings
    split_dataset_ratio: Optional[float] = None
    max_length: Optional[int] = None
    padding_free: bool = False
    # Task type
    task_type: Optional[str] = None
    loss_type: Optional[str] = None
    num_labels: Optional[int] = None
    use_chat_template: Optional[bool] = None
    # Self-cognition
    model_name: Optional[str] = None
    model_author: Optional[str] = None
    # Hub save
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    hub_strategy: Optional[str] = None
    hub_token: Optional[str] = None
    # Reporting
    report_to: Optional[str] = None
    swanlab_token: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_workspace: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_mode: Optional[str] = None


class RLHFRequest(TrainRequest):
    rlhf_type: Optional[str] = None
    ref_model: Optional[str] = None
    ref_model_type: Optional[str] = None
    reward_model: Optional[str] = None
    reward_model_type: Optional[str] = None
    teacher_model: Optional[str] = None
    teacher_model_type: Optional[str] = None
    beta: Optional[float] = None
    max_completion_length: Optional[int] = None
    loss_scale: Optional[str] = None
    lmbda: Optional[float] = None
    cpo_alpha: Optional[float] = None
    rpo_alpha: Optional[float] = None
    simpo_gamma: Optional[float] = None
    desirable_weight: Optional[float] = None
    undesirable_weight: Optional[float] = None


class GRPORequest(RLHFRequest):
    vllm_mode: Optional[str] = None
    num_generations: Optional[int] = None
    reward_funcs: Optional[List[str]] = None
    reward_weights: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    vllm_gpu_memory_utilization: Optional[str] = None
    vllm_tensor_parallel_size: Optional[int] = None
    vllm_max_model_len: Optional[int] = None
    vllm_server_host: Optional[str] = None
    vllm_server_port: Optional[int] = None
    vllm_server_timeout: Optional[int] = None
    epsilon: Optional[float] = None
    epsilon_high: Optional[float] = None
    num_iterations: Optional[int] = None


class InferRequest(BaseModel):
    model: str
    model_type: Optional[str] = None
    template: Optional[str] = None
    adapters: Optional[str] = None
    merge_lora: bool = False
    infer_backend: str = 'transformers'
    port: int = 8000
    gpu_ids: Optional[List[str]] = None
    more_params: Optional[str] = None


class TensorBoardStartRequest(BaseModel):
    logging_dir: str


class ChatRequest(BaseModel):
    port: int = 8000
    model: Optional[str] = None          # actual model path/ID used when deploying
    messages: List[Dict[str, Any]]
    system: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.7
    repetition_penalty: float = 1.05
    stream: bool = False
    model_name: Optional[str] = None     # optional LoRA module name override


class ExportRequest(BaseModel):
    model: str
    model_type: Optional[str] = None
    template: Optional[str] = None
    merge_lora: bool = False
    quant_bits: Optional[int] = None
    quant_method: Optional[str] = None
    quant_n_samples: Optional[int] = None
    max_length: Optional[int] = None
    output_dir: Optional[str] = None
    dataset: Optional[List[str]] = None
    gpu_ids: Optional[List[str]] = None
    device_map: Optional[str] = None
    more_params: Optional[str] = None


class EvalRequest(BaseModel):
    model: str
    model_type: Optional[str] = None
    template: Optional[str] = None
    eval_backend: Optional[str] = None
    eval_dataset: Optional[List[str]] = None
    eval_limit: Optional[int] = None
    infer_backend: Optional[str] = None
    custom_eval_config: Optional[str] = None
    eval_output_dir: Optional[str] = None
    eval_url: Optional[str] = None
    api_key: Optional[str] = None
    gpu_ids: Optional[List[str]] = None
    more_params: Optional[str] = None


class SampleRequest(BaseModel):
    model: str
    model_type: Optional[str] = None
    template: Optional[str] = None
    dataset: Optional[List[str]] = None
    system: Optional[str] = None
    sampler_type: Optional[str] = None
    sampler_engine: Optional[str] = None
    num_return_sequences: Optional[int] = None
    num_sampling_batch_size: Optional[int] = None
    num_sampling_batches: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    prm_model: Optional[str] = None
    orm_model: Optional[str] = None
    n_best_to_keep: Optional[int] = None
    output_dir: str = 'sample_output'
    gpu_ids: Optional[List[str]] = None
    more_params: Optional[str] = None


class CmdPreviewRequest(BaseModel):
    action: str  # train / rlhf / grpo / infer / export / eval / sample
    params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Swift dataclass defaults (RLHFArguments / TrainArgumentsMixin / TunerArguments).
# Parameters whose value matches a Swift default are omitted from the CLI,
# mirroring Gradio's compare_value != value logic in LLMTrain.train().
SWIFT_TRAIN_DEFAULTS: Dict[str, Any] = {
    'seed': 42,
    'tuner_type': 'lora',
    'per_device_train_batch_size': 1,
    'per_device_eval_batch_size': 1,
    'num_train_epochs': 3,
    'save_steps': 500,
    'logging_steps': 5,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0,
    'weight_decay': 0.1,
    'max_grad_norm': 1,
    'sequence_parallel_size': 1,
    'split_dataset_ratio': 0.0,
    'lora_rank': 8,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'target_modules': 'all-linear',
    'tuner_backend': 'peft',
    'report_to': 'tensorboard',
}

RLHF_DEFAULTS: Dict[str, Any] = {
    'rlhf_type': 'dpo',
    'max_completion_length': 512,
    'cpo_alpha': 1.0,
    'simpo_gamma': 1,
    'desirable_weight': 1.0,
    'undesirable_weight': 1.0,
    'lmbda': 0.5,
}

GRPO_DEFAULTS: Dict[str, Any] = {
    'num_generations': 8,
    'num_iterations': 1,
    'temperature': 0.9,
    'top_k': -1,
    'top_p': 1.0,
    'repetition_penalty': 1.0,
    'epsilon': 0.2,
    'vllm_gpu_memory_utilization': 0.9,
    'vllm_tensor_parallel_size': 1,
    'vllm_server_timeout': 240,
}

RLHF_TYPE_ARGS: Dict[str, set] = {
    'dpo': {'rpo_alpha', 'ref_model', 'ref_model_type'},
    'cpo': {'cpo_alpha'},
    'kto': {'desirable_weight', 'undesirable_weight', 'ref_model', 'ref_model_type'},
    'simpo': {'simpo_gamma', 'cpo_alpha'},
    'gkd': {'teacher_model', 'teacher_model_type', 'max_completion_length', 'lmbda'},
    'ppo': {'reward_model', 'reward_model_type', 'max_completion_length', 'ref_model', 'ref_model_type'},
}


def _timestamp() -> str:
    now = datetime.now()
    return f'{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}'


# Clear all accelerator visibility masks so a child process does not inherit the
# parent Web-UI's CUDA/ASCEND/MLU devices when the user explicitly selects CPU.
_CPU_CLEAR_ENV = {
    'CUDA_VISIBLE_DEVICES': '',
    # torch_npu rejects an empty ASCEND_RT_VISIBLE_DEVICES value. None tells
    # the subprocess launcher to remove an inherited mask instead.
    'ASCEND_RT_VISIBLE_DEVICES': None,
    'MLU_VISIBLE_DEVICES': '',
}

# Cache durable device-list results to avoid intermittent empty GPU dropdowns on
# NPU (torch.npu.device_count can briefly fail or return 0 under concurrent load).
_DEVICE_INFO_CACHE: Optional[tuple] = None


def _build_gpu_env(gpu_ids: Optional[List[str]]) -> Dict[str, Optional[str]]:
    if not gpu_ids:
        return {}
    gpu_ids = [g for g in gpu_ids if g]
    if not gpu_ids or 'cpu' in gpu_ids:
        return dict(_CPU_CLEAR_ENV)
    # Single-device UI labels (gpu/npu/mlu) always map to physical index 0.
    if len(gpu_ids) == 1 and gpu_ids[0] in ('gpu', 'npu', 'mlu'):
        gpu_ids = ['0']
    gpus = ','.join(gpu_ids)
    try:
        if is_torch_npu_available():
            return {'ASCEND_RT_VISIBLE_DEVICES': gpus}
        elif is_torch_cuda_available():
            return {'CUDA_VISIBLE_DEVICES': gpus}
        elif is_torch_mlu_available():
            return {'MLU_VISIBLE_DEVICES': gpus}
    except Exception:
        pass
    return {'CUDA_VISIBLE_DEVICES': gpus}


def _safe_device_count() -> int:
    """Probe accelerator count with short retries (NPU can briefly report 0)."""
    last_err: Optional[BaseException] = None
    for attempt in range(3):
        try:
            count = get_device_count()
            if count > 0:
                return count
        except Exception as e:
            last_err = e
            logger.warning('get_device_count failed (attempt %s): %s', attempt + 1, e)
        if attempt < 2:
            time.sleep(0.05)
    if last_err is not None:
        logger.warning('get_device_count ultimately failed, treating as 0: %s', last_err)
    return 0


def _probe_restful_ui_device_info() -> tuple[List[str], str, bool]:
    """Return (choices, default, durable).

    ``durable`` is True when the runtime reported a positive device count (safe to
    cache). Transient 0-count fallbacks still return accelerator labels so the UI
    is not empty, but are not cached.
    """
    device_count = _safe_device_count()
    durable = device_count > 0
    try:
        npu = is_torch_npu_available()
    except Exception:
        npu = False
    try:
        cuda = is_torch_cuda_available()
    except Exception:
        cuda = False
    try:
        mlu = is_torch_mlu_available()
    except Exception:
        mlu = False

    # Prefer accelerator-type labels even when runtime count briefly reports 0,
    # so the UI dropdown is not empty after refresh on NPU/GPU hosts.
    if npu:
        if device_count <= 1:
            return ['npu', 'cpu'], 'npu', durable
        return [str(i) for i in range(device_count)] + ['cpu'], '0', durable
    if cuda:
        if device_count <= 1:
            return ['gpu', 'cpu'], 'gpu', durable
        return [str(i) for i in range(device_count)] + ['cpu'], '0', durable
    if mlu:
        if device_count <= 1:
            return ['mlu', 'cpu'], 'mlu', durable
        return [str(i) for i in range(device_count)] + ['cpu'], '0', durable
    choices, default = get_ui_device_info()
    return choices, default, True


def _get_restful_ui_device_info() -> tuple[List[str], str]:
    global _DEVICE_INFO_CACHE
    if _DEVICE_INFO_CACHE is not None:
        return _DEVICE_INFO_CACHE
    choices, default, durable = _probe_restful_ui_device_info()
    info = (choices, default)
    if durable and choices:
        _DEVICE_INFO_CACHE = info
    return info


def _parse_more_params(more_params: Optional[str]) -> tuple:
    """Returns (json_dict, cmd_str)."""
    if not more_params or not more_params.strip():
        return {}, ''
    try:
        return json.loads(more_params), ''
    except Exception:
        return {}, more_params.strip()


def _add_more_params_to_cmd(command: list, more_params_cmd: str):
    if not more_params_cmd:
        return
    parts = [p.strip() for p in more_params_cmd.split('--') if p.strip()]
    for part in parts:
        sub = part.split(' ', 1)
        command.append(f'--{sub[0]}')
        if len(sub) > 1 and sub[1].strip():
            command.append(sub[1].strip())


def _build_train_command(req: TrainRequest, subcmd: str, extra_flags: Optional[list] = None):
    """Build swift train command list, envs dict, log_file path."""
    command = ['swift', subcmd]
    if extra_flags:
        command.extend(extra_flags)

    def _add(key, value):
        if value is None:
            return
        if isinstance(value, list):
            command.append(f'--{key}')
            command.extend([str(v) for v in value])
        elif isinstance(value, bool):
            command.extend([f'--{key}', str(value).lower()])
        else:
            command.extend([f'--{key}', str(value)])

    def _add_if_changed(key, value):
        """Only emit the flag when the value differs from Swift's dataclass default."""
        if value is None:
            return
        swift_default = SWIFT_TRAIN_DEFAULTS.get(key)
        if swift_default is not None and value == swift_default:
            return
        _add(key, value)

    _add('model', req.model)
    if req.model_type:
        _add('model_type', req.model_type)
    if req.template:
        _add('template', req.template)
    if req.dataset:
        _add('dataset', req.dataset)
    _add_if_changed('tuner_type', req.tuner_type)
    _add_if_changed('seed', req.seed)
    if req.torch_dtype:
        _add('torch_dtype', req.torch_dtype)
    if req.use_liger_kernel:
        _add('use_liger_kernel', 'true')
    if req.deepspeed:
        _add('deepspeed', req.deepspeed)
    _add_if_changed('sequence_parallel_size', req.sequence_parallel_size)
    if req.learning_rate is not None:
        _add('learning_rate', req.learning_rate)
    _add_if_changed('per_device_train_batch_size', req.per_device_train_batch_size)
    _add_if_changed('per_device_eval_batch_size', req.per_device_eval_batch_size)
    _add_if_changed('num_train_epochs', req.num_train_epochs)
    if req.eval_steps:
        _add('eval_steps', req.eval_steps)
    _add_if_changed('save_steps', req.save_steps)
    if req.gradient_accumulation_steps:
        _add('gradient_accumulation_steps', req.gradient_accumulation_steps)
    if req.attn_impl:
        _add('attn_impl', req.attn_impl)
    if req.neftune_noise_alpha:
        _add('neftune_noise_alpha', req.neftune_noise_alpha)
    if req.system:
        _add('system', req.system)
    # Dataset settings
    _add_if_changed('split_dataset_ratio', req.split_dataset_ratio)
    if req.max_length:
        _add('max_length', req.max_length)
    if req.padding_free:
        _add('padding_free', 'true')
    # Advanced settings
    _add_if_changed('tuner_backend', req.tuner_backend)
    _add_if_changed('weight_decay', req.weight_decay)
    _add_if_changed('logging_steps', req.logging_steps)
    _add_if_changed('lr_scheduler_type', req.lr_scheduler_type)
    _add_if_changed('warmup_ratio', req.warmup_ratio)
    if req.truncation_strategy:
        _add('truncation_strategy', req.truncation_strategy)
    if req.max_steps is not None and req.max_steps > 0:
        _add('max_steps', req.max_steps)
    _add_if_changed('max_grad_norm', req.max_grad_norm)
    # LoRA params — skip when tuner_type=full (Gradio remove_useless_args parity)
    _tuner = req.tuner_type or 'lora'
    if _tuner != 'full':
        _add_if_changed('lora_rank', req.lora_rank)
        _add_if_changed('lora_alpha', req.lora_alpha)
        _add_if_changed('lora_dropout', req.lora_dropout)
        if req.lora_dtype:
            _add('lora_dtype', req.lora_dtype)
        if req.use_rslora:
            _add('use_rslora', 'true')
        if req.use_dora:
            _add('use_dora', 'true')
        _add_if_changed('target_modules', req.target_modules)
    # Task type
    if req.task_type:
        _add('task_type', req.task_type)
    if req.loss_type:
        _add('loss_type', req.loss_type)
    if req.num_labels is not None:
        _add('num_labels', str(req.num_labels))
    if req.use_chat_template is not None:
        _add('use_chat_template', str(req.use_chat_template).lower())
    # Self-cognition
    if req.model_name:
        _add('model_name', req.model_name)
    if req.model_author:
        _add('model_author', req.model_author)
    # Hub save
    if req.push_to_hub:
        _add('push_to_hub', 'true')
    if req.hub_model_id:
        _add('hub_model_id', req.hub_model_id)
    if req.hub_private_repo:
        _add('hub_private_repo', 'true')
    if req.hub_strategy:
        _add('hub_strategy', req.hub_strategy)
    if req.hub_token:
        _add('hub_token', req.hub_token)
    # Reporting
    _add_if_changed('report_to', req.report_to)
    if req.swanlab_token:
        _add('swanlab_token', req.swanlab_token)
    if req.swanlab_project:
        _add('swanlab_project', req.swanlab_project)
    if req.swanlab_workspace:
        _add('swanlab_workspace', req.swanlab_workspace)
    if req.swanlab_exp_name:
        _add('swanlab_exp_name', req.swanlab_exp_name)
    if req.swanlab_mode:
        _add('swanlab_mode', req.swanlab_mode)

    # output/logging dirs
    model_type = req.model_type or 'model'
    ts = _timestamp()
    logging_dir = req.logging_dir or f'output/{model_type}-{ts}'
    output_dir = req.output_dir or logging_dir
    command.extend(['--add_version', 'False', '--output_dir', output_dir,
                    '--logging_dir', logging_dir, '--ignore_args_error', 'True'])

    # more_params
    mp_dict, mp_cmd = _parse_more_params(req.more_params)
    for k, v in mp_dict.items():
        _add(k, v)
    _add_more_params_to_cmd(command, mp_cmd)
    # Hiding accelerators is not enough for Transformers: its training
    # arguments must also be told explicitly to initialize a CPU device.
    # Add this last so custom more_params cannot contradict the UI selection.
    if req.gpu_ids and 'cpu' in req.gpu_ids:
        _add('use_cpu', True)

    # log file
    os.makedirs(logging_dir, exist_ok=True)
    log_file = os.path.join(os.getcwd(), logging_dir, 'run.log')
    command.extend(['--log_file', log_file])

    # Normalize logging_dir to absolute so the response always carries an absolute path
    logging_dir = os.path.abspath(logging_dir)

    # envs
    all_envs = _build_gpu_env(req.gpu_ids)
    if req.use_ddp and req.ddp_num > 1:
        all_envs['NPROC_PER_NODE'] = str(req.ddp_num)
    if req.envs:
        for item in req.envs.strip().split():
            if '=' in item:
                k, v = item.split('=', 1)
                all_envs[k] = v

    return command, all_envs, log_file, logging_dir


def _build_simple_command(subcmd: str, model: str, model_type: Optional[str],
                           gpu_ids: Optional[List[str]], more_params: Optional[str],
                           extra_kwargs: Optional[Dict] = None) -> tuple:
    """Build a simple swift command (export/eval/sample)."""
    command = ['swift', subcmd]
    command.extend(['--model', model])
    if model_type:
        command.extend(['--model_type', model_type])

    if extra_kwargs:
        for k, v in extra_kwargs.items():
            if v is None:
                continue
            if isinstance(v, list):
                command.append(f'--{k}')
                command.extend([str(i) for i in v])
            elif isinstance(v, bool):
                command.extend([f'--{k}', str(v).lower()])
            else:
                command.extend([f'--{k}', str(v)])

    mp_dict, mp_cmd = _parse_more_params(more_params)
    for k, v in mp_dict.items():
        if isinstance(v, list):
            command.append(f'--{k}')
            command.extend([str(i) for i in v])
        else:
            command.extend([f'--{k}', str(v)])
    _add_more_params_to_cmd(command, mp_cmd)

    mt = model_type or 'model'
    ts = _timestamp()
    log_path = f'output/{mt}-{ts}/run_{subcmd}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = os.path.join(os.getcwd(), log_path)
    command.extend(['--log_file', log_file, '--ignore_args_error', 'true'])

    all_envs = _build_gpu_env(gpu_ids)
    return command, all_envs, log_file


# ---------------------------------------------------------------------------
# Training records (file-based cache, same approach as Gradio BaseUI)
# ---------------------------------------------------------------------------

def _get_records_dir() -> str:
    try:
        from modelscope.hub.utils.utils import get_cache_dir
        base = get_cache_dir()
    except Exception:
        base = os.path.expanduser('~/.cache/modelscope')
    records_dir = os.path.join(base, 'swift-web-ui')
    os.makedirs(records_dir, exist_ok=True)
    return records_dir


def _model_key(model: str) -> str:
    return model.replace('/', '-')


def _record_filename_prefix(scope: str, model: str) -> str:
    return f'{scope}--{_model_key(model)}'


def _save_scoped_record(scope: str, model: str, params: dict) -> str:
    ts = str(int(time.time()))
    prefix = _record_filename_prefix(scope, model)
    filename = os.path.join(_get_records_dir(), f'{prefix}-{ts}')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False)
    return ts


def _list_scoped_records(scope: str, model: str) -> List[str]:
    prefix = _record_filename_prefix(scope, model)
    records_dir = _get_records_dir()
    results = []
    for fname in os.listdir(records_dir):
        if fname.startswith(prefix + '-'):
            suffix = fname[len(prefix) + 1:]
            if suffix.isdigit():
                dt = datetime.fromtimestamp(int(suffix))
                results.append(dt.strftime('%Y/%m/%d %H:%M:%S'))
    return sorted(results, reverse=True)


def _load_scoped_record(scope: str, model: str, timestamp: str) -> dict:
    prefix = _record_filename_prefix(scope, model)
    records_dir = _get_records_dir()
    dt = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
    ts = str(int(dt.timestamp()))
    filename = os.path.join(records_dir, f'{prefix}-{ts}')
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def _delete_scoped_records(scope: str, model: str):
    prefix = _record_filename_prefix(scope, model)
    records_dir = _get_records_dir()
    for fname in os.listdir(records_dir):
        if fname.startswith(prefix + '-'):
            suffix = fname[len(prefix) + 1:]
            if suffix.isdigit():
                os.remove(os.path.join(records_dir, fname))


def _save_train_record(model: str, params: dict) -> str:
    return _save_scoped_record('train', model, params)


def _list_train_records(model: str) -> List[str]:
    return _list_scoped_records('train', model)


def _load_train_record(model: str, timestamp: str) -> dict:
    return _load_scoped_record('train', model, timestamp)


def _delete_train_records(model: str):
    _delete_scoped_records('train', model)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _default_thread_pool_workers() -> int:
    """Upper bound for blocking work offloaded via asyncio.to_thread / default executor."""
    return min(32, (os.cpu_count() or 1) * 5)


def create_app(
    thread_pool_workers: Optional[int] = None,
    tensorboard_path_prefix: Optional[str] = None,
) -> FastAPI:
    from .tensorboard_manager import (filter_request_headers, filter_response_headers, get_tensorboard_backend_origin,
                                      get_tensorboard_upstream_base, normalize_path_prefix, start_tensorboard,
                                      stop_tensorboard, tensorboard_status)

    tb_prefix = normalize_path_prefix(
        tensorboard_path_prefix or os.environ.get('WEBUI_TENSORBOARD_PREFIX') or '/tensorboard')

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        n = thread_pool_workers if thread_pool_workers is not None else _default_thread_pool_workers()
        executor = ThreadPoolExecutor(max_workers=n)
        asyncio.get_running_loop().set_default_executor(executor)
        try:
            yield
        finally:
            stop_tensorboard()
            executor.shutdown(wait=False)

    app = FastAPI(
        title='SWIFT Web-UI Restful',
        version=getattr(swift, '__version__', ''),
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @app.get('/health')
    async def health():
        device_choices, default_device = await asyncio.to_thread(_get_restful_ui_device_info)
        try:
            version = swift.__version__
        except AttributeError:
            version = ''
        return {
            'status': 'ok',
            'version': version,
            'devices': device_choices,
            'default_device': default_device,
            'tensorboard_path_prefix': tb_prefix,
        }

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------
    @app.get('/api/v1/devices')
    async def devices():
        # Offload NPU/CUDA probes off the event loop; concurrent page-refresh
        # loads previously raced with heavy /models|/datasets imports.
        device_choices, default_device = await asyncio.to_thread(_get_restful_ui_device_info)
        return {'devices': device_choices, 'default': default_device}

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    @app.get('/api/v1/models')
    async def models():
        try:
            from swift.model import get_model_list
            model_list = get_model_list()
        except Exception:
            try:
                from swift.model import MODEL_MAPPING
                model_list = list(MODEL_MAPPING.keys())
            except Exception:
                model_list = []
        return {'models': model_list}

    @app.get('/api/v1/model-types')
    async def model_types():
        try:
            from swift.model import ModelType
            type_list = ModelType.get_model_name_list()
        except Exception:
            type_list = []
        return {'model_types': type_list}

    @app.get('/api/v1/templates')
    async def templates():
        try:
            from swift.template import TEMPLATE_MAPPING
            tmpl_list = sorted(TEMPLATE_MAPPING.keys())
        except Exception:
            tmpl_list = []
        return {'templates': tmpl_list}

    @app.get('/api/v1/model-meta')
    async def get_model_meta(model: str = Query('')):
        if not model:
            return {'template': '', 'model_type': '', 'system': '', 'matched_source': 'none'}
        model = model.strip()
        # Prefer local resume args (align with gradio update_input_model behavior).
        try:
            model_dir = os.path.expanduser(model)
            if not os.path.isabs(model_dir):
                model_dir = os.path.abspath(model_dir)
            args_file = os.path.join(model_dir, 'args.json')
            if os.path.isdir(model_dir) and os.path.isfile(args_file):
                with open(args_file, 'r', encoding='utf-8') as f:
                    args_data = json.load(f)
                template = args_data.get('template') or ''
                model_type = args_data.get('model_type') or ''
                system = args_data.get('system') or ''
                return {
                    'template': template,
                    'model_type': model_type,
                    'system': system,
                    'matched_source': 'args_json',
                }
        except Exception:
            pass
        try:
            from swift.model import get_matched_model_meta
            meta = get_matched_model_meta(model)
            if meta is not None:
                template = getattr(meta, 'template', '') or ''
                model_type = getattr(meta, 'model_type', '') or ''
                system = ''
                if template:
                    try:
                        from swift.template import TEMPLATE_MAPPING
                        tmpl = TEMPLATE_MAPPING.get(template)
                        if tmpl and hasattr(tmpl, 'default_system'):
                            system = tmpl.default_system or ''
                    except Exception:
                        pass
                return {'template': template, 'model_type': model_type, 'system': system, 'matched_source': 'registry'}
        except Exception:
            pass
        return {'template': '', 'model_type': '', 'system': '', 'matched_source': 'none'}

    @app.get('/api/v1/datasets')
    async def datasets():
        try:
            from swift.dataset import get_dataset_list
            ds_list = get_dataset_list()
        except Exception:
            ds_list = []
        return {'datasets': ds_list}

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    @app.get('/api/v1/tasks')
    async def tasks(cmd: Optional[str] = Query(None), rlhf_mode: Optional[str] = Query(None)):
        if cmd:
            if ',' in cmd:
                seen_log = set()
                merged: List[Dict] = []
                for part in cmd.split(','):
                    key = part.strip()
                    if not key:
                        continue
                    sub = await asyncio.to_thread(get_running_tasks, key)
                    for t in sub:
                        lf = t.get('log_file')
                        if lf:
                            if lf in seen_log:
                                continue
                            seen_log.add(lf)
                        merged.append(t)
                task_list = merged
            else:
                task_list = await asyncio.to_thread(get_running_tasks, cmd)
        else:
            task_list = await asyncio.to_thread(get_all_running_tasks)
        if rlhf_mode and task_list:
            filtered: List[Dict] = []
            for t in task_list:
                if t.get('cmd') != 'rlhf':
                    filtered.append(t)
                    continue
                task_type = ((t.get('args') or {}).get('rlhf_type') or '').strip()
                is_grpo = task_type == 'grpo'
                if rlhf_mode == 'grpo' and is_grpo:
                    filtered.append(t)
                elif rlhf_mode == 'non_grpo' and not is_grpo:
                    filtered.append(t)
            task_list = filtered
        return {'tasks': task_list}

    @app.delete('/api/v1/tasks')
    async def stop_task(log_file: str = Query(..., description='Log file path of the task to kill')):
        ok = await asyncio.to_thread(kill_task_by_log, log_file)
        if not ok:
            raise HTTPException(status_code=404, detail='Task not found or already stopped')
        return {'status': 'killed', 'log_file': log_file}

    # ------------------------------------------------------------------
    # Log streaming (SSE)
    # ------------------------------------------------------------------
    @app.get('/api/v1/log')
    async def stream_log(path: str = Query(..., description='Absolute log file path')):
        async def _event_generator():
            # Use the app default ThreadPoolExecutor (same pool as asyncio.to_thread).
            loop = asyncio.get_running_loop()
            gen = tail_log_file(path)
            # PEP 479: a bare `StopIteration` raised inside a Future (which is
            # what `loop.run_in_executor(None, next, gen)` produces when the
            # generator is exhausted) cannot be propagated and triggers a
            # noisy "StopIteration interacts badly with generators" error.
            # Use a sentinel so the worker returns a normal value on EOF.
            _SENTINEL = object()

            def _next_chunk():
                return next(gen, _SENTINEL)

            while True:
                try:
                    text = await loop.run_in_executor(None, _next_chunk)
                except Exception:
                    break
                if text is _SENTINEL:
                    break
                yield f'data: {json.dumps({"text": text})}\n\n'

        return StreamingResponse(_event_generator(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    # ------------------------------------------------------------------
    # Train metrics (TensorBoard scalars, aligned with Gradio Runtime.sft_plot)
    # ------------------------------------------------------------------
    @app.get('/api/v1/train/tensorboard-metrics')
    async def train_tensorboard_metrics(logging_dir: str = Query(..., description='Training logging_dir path')):
        from .tb_series import build_sft_pt_series

        def _run():
            return build_sft_pt_series(logging_dir)

        return await asyncio.to_thread(_run)

    @app.get('/api/v1/rlhf/tensorboard-metrics')
    async def rlhf_tensorboard_metrics(
        logging_dir: str = Query(..., description='RLHF logging_dir path'),
        rlhf_type: str = Query('dpo', description='RLHF type (dpo/kto/orpo/...)'),
    ):
        from .tb_series import build_rlhf_series

        def _run():
            return build_rlhf_series(logging_dir, rlhf_type)

        return await asyncio.to_thread(_run)

    @app.get('/api/v1/grpo/tensorboard-metrics')
    async def grpo_tensorboard_metrics(logging_dir: str = Query(..., description='GRPO logging_dir path')):
        from .tb_series import build_grpo_series

        def _run():
            return build_grpo_series(logging_dir)

        return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # Training records
    # ------------------------------------------------------------------

    @app.get('/api/v1/train/records')
    async def train_records_list(model: str = Query(...)):
        try:
            records = await asyncio.to_thread(_list_train_records, model)
            return {'records': records}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/v1/train/records/detail')
    async def train_records_detail(model: str = Query(...), timestamp: str = Query(...)):
        try:
            params = await asyncio.to_thread(_load_train_record, model, timestamp)
            return params
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail='Record not found')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete('/api/v1/train/records')
    async def train_records_delete(model: str = Query(...)):
        try:
            await asyncio.to_thread(_delete_train_records, model)
            return {'status': 'ok'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/v1/rlhf/records')
    async def rlhf_records_list(model: str = Query(...)):
        try:
            records = await asyncio.to_thread(_list_scoped_records, 'rlhf', model)
            return {'records': records}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/v1/rlhf/records/detail')
    async def rlhf_records_detail(model: str = Query(...), timestamp: str = Query(...)):
        try:
            return await asyncio.to_thread(_load_scoped_record, 'rlhf', model, timestamp)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail='Record not found')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete('/api/v1/rlhf/records')
    async def rlhf_records_delete(model: str = Query(...)):
        try:
            await asyncio.to_thread(_delete_scoped_records, 'rlhf', model)
            return {'status': 'ok'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/v1/grpo/records')
    async def grpo_records_list(model: str = Query(...)):
        try:
            records = await asyncio.to_thread(_list_scoped_records, 'grpo', model)
            return {'records': records}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/api/v1/grpo/records/detail')
    async def grpo_records_detail(model: str = Query(...), timestamp: str = Query(...)):
        try:
            return await asyncio.to_thread(_load_scoped_record, 'grpo', model, timestamp)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail='Record not found')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete('/api/v1/grpo/records')
    async def grpo_records_delete(model: str = Query(...)):
        try:
            await asyncio.to_thread(_delete_scoped_records, 'grpo', model)
            return {'status': 'ok'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # TensorBoard subprocess + reverse proxy under tb_prefix
    # ------------------------------------------------------------------
    @app.get('/api/v1/tensorboard/status')
    async def tensorboard_status_api():
        return await asyncio.to_thread(tensorboard_status)

    @app.post('/api/v1/tensorboard/start')
    async def tensorboard_start_api(req: TensorBoardStartRequest, request: Request):
        def _run():
            return start_tensorboard(req.logging_dir, tb_prefix)

        ok, msg, port, url_suffix = await asyncio.to_thread(_run)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {
            'status': 'ok',
            'message': msg,
            'port': port,
            'url': url_suffix,
        }

    @app.post('/api/v1/tensorboard/stop')
    async def tensorboard_stop_api():
        await asyncio.to_thread(stop_tensorboard)
        return {'status': 'stopped'}

    async def _tensorboard_http_forward(request: Request, target_url: str):
        """Proxy to local TensorBoard; avoid gzip header/body mismatch (httpx decodes but TB sends Content-Encoding)."""
        try:
            import httpx
        except ImportError:
            raise HTTPException(status_code=500, detail='httpx is required for TensorBoard proxy')
        hdrs = filter_request_headers(request.headers)
        hdrs['accept-encoding'] = 'identity'
        body = await request.body()
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            r = await client.request(
                request.method,
                target_url,
                headers=hdrs,
                content=body if body else None,
                follow_redirects=False,
            )
        out = await r.aread()
        rh = filter_response_headers(r.headers)
        for _k in list(rh.keys()):
            if _k.lower() in ('content-length', 'content-encoding', 'transfer-encoding'):
                del rh[_k]
        return Response(content=out, status_code=r.status_code, headers=rh)

    async def _tensorboard_try_urls(request: Request, urls: List[str]):
        """Try proxy URLs in order; use first non-404 response (for TB root vs path_prefix asset layouts)."""
        try:
            import httpx
        except ImportError:
            raise HTTPException(status_code=500, detail='httpx is required for TensorBoard proxy')
        hdrs = filter_request_headers(request.headers)
        hdrs['accept-encoding'] = 'identity'
        body = await request.body()
        r = None
        out = b''
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            for url in urls:
                r = await client.request(
                    request.method,
                    url,
                    headers=hdrs,
                    content=body if body else None,
                    follow_redirects=False,
                )
                out = await r.aread()
                if r.status_code != 404:
                    break
        assert r is not None
        rh = filter_response_headers(r.headers)
        for _k in list(rh.keys()):
            if _k.lower() in ('content-length', 'content-encoding', 'transfer-encoding'):
                del rh[_k]
        return Response(content=out, status_code=r.status_code, headers=rh)

    # TensorBoard still references some assets from site root (e.g. /font-roboto/) even with --path_prefix;
    # forward those to the same backend process.
    @app.api_route(
        '/font-roboto/{path:path}',
        methods=['GET', 'POST', 'HEAD', 'OPTIONS'],
        include_in_schema=False,
    )
    async def _tensorboard_font_roboto_proxy(request: Request, path: str):
        origin = get_tensorboard_backend_origin()
        if not origin:
            raise HTTPException(status_code=404, detail='TensorBoard is not running')
        tail = path.strip('/')
        path_part = f'font-roboto/{tail}' if tail else 'font-roboto'
        q = (('?' + str(request.url.query)) if request.url.query else '')
        urls = [
            f'{origin}/{path_part}{q}',
            f'{origin}{tb_prefix}/{path_part}{q}',
        ]
        return await _tensorboard_try_urls(request, urls)

    @app.api_route(
        '/data/{path:path}',
        methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD', 'PATCH'],
        include_in_schema=False,
    )
    async def _tensorboard_data_root_proxy(request: Request, path: str):
        origin = get_tensorboard_backend_origin()
        if not origin:
            raise HTTPException(status_code=404, detail='TensorBoard is not running')
        tail = path.strip('/')
        path_part = f'data/{tail}' if tail else 'data'
        q = (('?' + str(request.url.query)) if request.url.query else '')
        urls = [
            f'{origin}/{path_part}{q}',
            f'{origin}{tb_prefix}/{path_part}{q}',
        ]
        return await _tensorboard_try_urls(request, urls)

    @app.get(tb_prefix, include_in_schema=False)
    async def _tensorboard_redirect_trailing_slash():
        return RedirectResponse(url=f'{tb_prefix}/')

    @app.api_route(
        f'{tb_prefix}/{{full_path:path}}',
        methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD', 'PATCH'],
        include_in_schema=False,
    )
    async def _tensorboard_http_proxy(request: Request, full_path: str):
        upstream_root = get_tensorboard_upstream_base()
        if not upstream_root:
            raise HTTPException(status_code=503, detail='TensorBoard is not running')
        tail = full_path.strip('/')
        url = upstream_root + tail + (('?' + str(request.url.query)) if request.url.query else '')
        return await _tensorboard_http_forward(request, url)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    @app.post('/api/v1/train/start')
    async def train_start(req: TrainRequest):
        subcmd = req.train_stage if req.train_stage in ('sft', 'pt') else 'sft'
        command, all_envs, log_file, logging_dir = _build_train_command(req, subcmd)
        if not req.dry_run:
            await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
            # auto-save training record (exclude sensitive/volatile fields)
            try:
                try:
                    record_params = req.model_dump(exclude={'dry_run', 'hub_token', 'swanlab_token'})
                except AttributeError:
                    record_params = req.dict(exclude={'dry_run', 'hub_token', 'swanlab_token'})
                # Persist resolved log_file / logging_dir so finished tasks can
                # still display their training log when the record is selected.
                record_params['log_file'] = log_file
                record_params['logging_dir'] = logging_dir
                await asyncio.to_thread(_save_train_record, req.model, record_params)
            except Exception:
                pass
        return {
            'status': 'started' if not req.dry_run else 'dry_run',
            'command': ' '.join(command),
            'log_file': log_file,
            'logging_dir': logging_dir,
        }

    # ------------------------------------------------------------------
    # RLHF
    # ------------------------------------------------------------------
    @app.post('/api/v1/rlhf/start')
    async def rlhf_start(req: RLHFRequest):
        command, all_envs, log_file, logging_dir = _build_train_command(req, 'rlhf')

        cur_type = req.rlhf_type or 'dpo'
        cur_type_args = RLHF_TYPE_ARGS.get(cur_type, set())
        other_type_args: set = set()
        for t, args in RLHF_TYPE_ARGS.items():
            if t != cur_type:
                other_type_args |= (args - cur_type_args)

        def _add_rlhf(key, value):
            if value is None:
                return
            if isinstance(value, str) and not value:
                return
            d = RLHF_DEFAULTS.get(key)
            if d is not None and value == d:
                return
            if key in other_type_args and value:
                return
            command.extend([f'--{key}', str(value)])

        _add_rlhf('rlhf_type', req.rlhf_type)
        _add_rlhf('ref_model', req.ref_model)
        _add_rlhf('ref_model_type', req.ref_model_type)
        _add_rlhf('reward_model', req.reward_model)
        _add_rlhf('reward_model_type', req.reward_model_type)
        _add_rlhf('teacher_model', req.teacher_model)
        _add_rlhf('teacher_model_type', req.teacher_model_type)
        _add_rlhf('beta', req.beta)
        _add_rlhf('max_completion_length', req.max_completion_length)
        _add_rlhf('loss_scale', req.loss_scale)
        _add_rlhf('lmbda', req.lmbda)
        _add_rlhf('cpo_alpha', req.cpo_alpha)
        _add_rlhf('rpo_alpha', req.rpo_alpha)
        _add_rlhf('simpo_gamma', req.simpo_gamma)
        _add_rlhf('desirable_weight', req.desirable_weight)
        _add_rlhf('undesirable_weight', req.undesirable_weight)
        if not req.dry_run:
            await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
            try:
                try:
                    record_params = req.model_dump(exclude={'dry_run', 'hub_token', 'swanlab_token'})
                except AttributeError:
                    record_params = req.dict(exclude={'dry_run', 'hub_token', 'swanlab_token'})
                record_params['log_file'] = log_file
                record_params['logging_dir'] = logging_dir
                await asyncio.to_thread(_save_scoped_record, 'rlhf', req.model, record_params)
            except Exception:
                pass
        return {
            'status': 'started' if not req.dry_run else 'dry_run',
            'command': ' '.join(command),
            'log_file': log_file,
            'logging_dir': logging_dir,
        }

    # ------------------------------------------------------------------
    # GRPO
    # ------------------------------------------------------------------
    @app.post('/api/v1/grpo/start')
    async def grpo_start(req: GRPORequest):
        command, all_envs, log_file, logging_dir = _build_train_command(req, 'rlhf')
        command.extend(['--rlhf_type', 'grpo'])

        def _add_grpo(key, value):
            if value is None:
                return
            if isinstance(value, str) and not value:
                return
            d = GRPO_DEFAULTS.get(key)
            if d is not None and value == d:
                return
            command.extend([f'--{key}', str(value)])

        if req.vllm_mode:
            command.extend(['--vllm_mode', req.vllm_mode])
        _add_grpo('num_generations', req.num_generations)
        if req.reward_funcs:
            command.append('--reward_funcs')
            command.extend(req.reward_funcs)
        if req.reward_weights:
            command.extend(['--reward_weights', req.reward_weights])
        if req.ref_model:
            command.extend(['--ref_model', req.ref_model])
        _add_grpo('beta', req.beta)
        _add_grpo('max_completion_length', req.max_completion_length)
        _add_grpo('temperature', req.temperature)
        _add_grpo('top_k', req.top_k)
        _add_grpo('top_p', req.top_p)
        _add_grpo('repetition_penalty', req.repetition_penalty)
        _add_grpo('vllm_gpu_memory_utilization', req.vllm_gpu_memory_utilization)
        _add_grpo('vllm_tensor_parallel_size', req.vllm_tensor_parallel_size)
        if req.vllm_max_model_len is not None:
            command.extend(['--vllm_max_model_len', str(req.vllm_max_model_len)])
        if req.vllm_server_host:
            command.extend(['--vllm_server_host', req.vllm_server_host])
        if req.vllm_server_port is not None:
            command.extend(['--vllm_server_port', str(req.vllm_server_port)])
        _add_grpo('vllm_server_timeout', req.vllm_server_timeout)
        _add_grpo('epsilon', req.epsilon)
        if req.epsilon_high is not None:
            command.extend(['--epsilon_high', str(req.epsilon_high)])
        _add_grpo('num_iterations', req.num_iterations)
        if not req.dry_run:
            await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
            try:
                try:
                    record_params = req.model_dump(exclude={'dry_run', 'hub_token', 'swanlab_token'})
                except AttributeError:
                    record_params = req.dict(exclude={'dry_run', 'hub_token', 'swanlab_token'})
                record_params['log_file'] = log_file
                record_params['logging_dir'] = logging_dir
                await asyncio.to_thread(_save_scoped_record, 'grpo', req.model, record_params)
            except Exception:
                pass
        return {
            'status': 'started' if not req.dry_run else 'dry_run',
            'command': ' '.join(command),
            'log_file': log_file,
            'logging_dir': logging_dir,
        }

    # ------------------------------------------------------------------
    # Infer (deploy)
    # ------------------------------------------------------------------
    @app.post('/api/v1/infer/start')
    async def infer_start(req: InferRequest):
        used_ports = await asyncio.to_thread(get_used_ports, 'deploy')
        if req.port in used_ports:
            raise HTTPException(status_code=409, detail=f'Port {req.port} is already in use')

        mt = req.model_type or 'model'
        ts = _timestamp()
        log_path = f'output/{mt}-{ts}/run_deploy.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = os.path.join(os.getcwd(), log_path)

        command = ['swift', 'deploy', '--model', req.model]
        if req.model_type:
            command.extend(['--model_type', req.model_type])
        if req.template:
            command.extend(['--template', req.template])
        if req.adapters:
            command.extend(['--adapters', req.adapters])
        if req.merge_lora:
            command.extend(['--merge_lora', 'true'])
        if req.infer_backend:
            command.extend(['--infer_backend', req.infer_backend])
        command.extend(['--port', str(req.port)])
        mp_dict, mp_cmd = _parse_more_params(req.more_params)
        for k, v in mp_dict.items():
            command.extend([f'--{k}', str(v)])
        _add_more_params_to_cmd(command, mp_cmd)
        command.extend(['--log_file', log_file, '--ignore_args_error', 'true'])

        all_envs = _build_gpu_env(req.gpu_ids)
        await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
        return {
            'status': 'started',
            'command': ' '.join(command),
            'log_file': log_file,
            'port': req.port,
        }

    # ------------------------------------------------------------------
    # Infer (chat) — calls the deployed server's OpenAI-compatible API
    # directly via httpx to avoid InferClient's event-loop threading conflict
    # with uvicorn's already-running asyncio loop.
    # ------------------------------------------------------------------
    @app.post('/api/v1/infer/chat')
    async def infer_chat(req: ChatRequest):
        try:
            import httpx
        except ImportError:
            raise HTTPException(status_code=500, detail='httpx is required for chat (pip install httpx)')

        messages = list(req.messages)
        if req.system:
            if not messages or messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': req.system})

        # The deploy server registers models by basename (e.g. "Qwen3.5-0.8B"),
        # so strip any leading path/org prefix from the model ID.
        _raw_model = req.model_name or req.model or 'default'
        resolved_model = os.path.basename(_raw_model.rstrip('/')) or _raw_model

        payload: Dict[str, Any] = {
            'model': resolved_model,
            'messages': messages,
            'max_tokens': req.max_new_tokens,
            'temperature': req.temperature,
            'top_p': req.top_p,
            'repetition_penalty': req.repetition_penalty,
            'stream': req.stream,
        }
        if req.top_k and req.top_k > 0:
            payload['top_k'] = req.top_k

        infer_url = f'http://localhost:{req.port}/v1/chat/completions'

        if req.stream:
            async def _stream_gen():
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                        async with client.stream('POST', infer_url, json=payload) as response:
                            if response.status_code != 200:
                                body = await response.aread()
                                yield f'data: {json.dumps({"error": body.decode()})}\n\n'
                                return
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith('data:'):
                                    data = line[5:].strip()
                                    if data == '[DONE]':
                                        yield 'data: [DONE]\n\n'
                                        return
                                    try:
                                        chunk = json.loads(data)
                                        delta = (chunk.get('choices') or [{}])[0] \
                                            .get('delta', {}).get('content', '')
                                        if delta:
                                            yield f'data: {json.dumps({"delta": delta})}\n\n'
                                    except Exception:
                                        pass
                except httpx.ConnectError:
                    yield f'data: {json.dumps({"error": f"Cannot connect to inference server on port {req.port}. Is the model deployed?"})}\n\n'
                except Exception as e:
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'

            return StreamingResponse(
                _stream_gen(),
                media_type='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )
        else:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    response = await client.post(infer_url, json=payload)
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code,
                                            detail=response.text)
                    data = response.json()
                    content = (data.get('choices') or [{}])[0] \
                        .get('message', {}).get('content', '')
                    return {
                        'content': content,
                        'messages': req.messages + [{'role': 'assistant', 'content': content}],
                    }
            except httpx.ConnectError:
                raise HTTPException(
                    status_code=503,
                    detail=f'Cannot connect to inference server on port {req.port}. Is the model deployed?',
                )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    @app.post('/api/v1/export/start')
    async def export_start(req: ExportRequest):
        extra = {}
        if req.template:
            extra['template'] = req.template
        if req.merge_lora:
            extra['merge_lora'] = 'true'
        if req.quant_bits:
            extra['quant_bits'] = req.quant_bits
        if req.quant_method:
            extra['quant_method'] = req.quant_method
        if req.quant_n_samples:
            extra['quant_n_samples'] = req.quant_n_samples
        if req.max_length:
            extra['max_length'] = req.max_length
        if req.output_dir:
            extra['output_dir'] = req.output_dir
        if req.dataset:
            extra['dataset'] = req.dataset
        if req.device_map:
            extra['device_map'] = req.device_map
        command, all_envs, log_file = _build_simple_command(
            'export', req.model, req.model_type, req.gpu_ids, req.more_params, extra)
        await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
        return {'status': 'started', 'command': ' '.join(command), 'log_file': log_file}

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    @app.post('/api/v1/eval/start')
    async def eval_start(req: EvalRequest):
        extra = {}
        if req.template:
            extra['template'] = req.template
        if req.eval_backend:
            extra['eval_backend'] = req.eval_backend
        if req.eval_dataset:
            extra['eval_dataset'] = req.eval_dataset
        if req.eval_limit:
            extra['eval_limit'] = req.eval_limit
        if req.infer_backend:
            extra['infer_backend'] = req.infer_backend
        if req.custom_eval_config:
            extra['custom_eval_config'] = req.custom_eval_config
        if req.eval_output_dir:
            extra['eval_output_dir'] = req.eval_output_dir
        if req.eval_url:
            extra['eval_url'] = req.eval_url
        if req.api_key:
            extra['api_key'] = req.api_key
        command, all_envs, log_file = _build_simple_command(
            'eval', req.model, req.model_type, req.gpu_ids, req.more_params, extra)
        await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
        return {'status': 'started', 'command': ' '.join(command), 'log_file': log_file}

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    @app.post('/api/v1/sample/start')
    async def sample_start(req: SampleRequest):
        extra = {}
        if req.template:
            extra['template'] = req.template
        if req.dataset:
            extra['dataset'] = req.dataset
        if req.system:
            extra['system'] = req.system
        if req.sampler_type:
            extra['sampler_type'] = req.sampler_type
        if req.sampler_engine:
            extra['sampler_engine'] = req.sampler_engine
        if req.num_return_sequences:
            extra['num_return_sequences'] = req.num_return_sequences
        if req.num_sampling_batch_size:
            extra['num_sampling_batch_size'] = req.num_sampling_batch_size
        if req.num_sampling_batches:
            extra['num_sampling_batches'] = req.num_sampling_batches
        if req.max_new_tokens:
            extra['max_new_tokens'] = req.max_new_tokens
        if req.temperature is not None:
            extra['temperature'] = req.temperature
        if req.top_k is not None:
            extra['top_k'] = req.top_k
        if req.top_p is not None:
            extra['top_p'] = req.top_p
        if req.repetition_penalty is not None:
            extra['repetition_penalty'] = req.repetition_penalty
        if req.prm_model:
            extra['prm_model'] = req.prm_model
        if req.orm_model:
            extra['orm_model'] = req.orm_model
        if req.n_best_to_keep is not None:
            extra['n_best_to_keep'] = req.n_best_to_keep
        extra['output_dir'] = req.output_dir
        command, all_envs, log_file = _build_simple_command(
            'sample', req.model, req.model_type, req.gpu_ids, req.more_params, extra)
        await asyncio.to_thread(run_command_in_background_with_popen, command, all_envs, log_file)
        return {'status': 'started', 'command': ' '.join(command), 'log_file': log_file}

    # ------------------------------------------------------------------
    # Command preview
    # ------------------------------------------------------------------
    @app.post('/api/v1/cmd/preview')
    async def cmd_preview(req: CmdPreviewRequest):
        """Return the shell command without executing it."""
        action = req.action
        params = req.params
        command_parts = ['swift', action]
        for k, v in params.items():
            if v is None or v == '' or v == [] or v is False:
                continue
            if isinstance(v, list):
                command_parts.append(f'--{k}')
                command_parts.extend([str(i) for i in v])
            elif isinstance(v, bool):
                command_parts.extend([f'--{k}', str(v).lower()])
            else:
                command_parts.extend([f'--{k}', str(v)])
        return {'command': ' '.join(command_parts)}

    # ------------------------------------------------------------------
    # Static files (frontend) — must be last
    # ------------------------------------------------------------------
    if FRONTEND_DIR.exists():
        app.mount('/', StaticFiles(directory=str(FRONTEND_DIR), html=True), name='frontend')

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def webui_restful_main(args=None):
    if args is None:
        # When called directly (e.g. `python -m swift.ui.web_ui_restful`),
        # fall back to reading sys.argv so CLI flags are honoured.
        args = sys.argv[1:]

    if isinstance(args, list):
        import argparse
        parser = argparse.ArgumentParser(prog='swift web-ui-restful')
        parser.add_argument('--server_name', default='0.0.0.0')
        parser.add_argument('--server_port', type=int, default=7861)
        parser.add_argument('--lang', default='zh')
        parser.add_argument(
            '--thread_pool_workers',
            type=int,
            default=None,
            help='Max threads for blocking I/O offload (default: min(32, cpu_count*5)). '
            'Overrides WEBUI_THREAD_POOL_WORKERS.',
        )
        parser.add_argument(
            '--tensorboard_path_prefix',
            default=None,
            help="URL path prefix for proxied TensorBoard (default: /tensorboard). Overrides WEBUI_TENSORBOARD_PREFIX.",
        )
        parsed = parser.parse_args(args)
        tpw = parsed.thread_pool_workers
        if tpw is None:
            tp_env = os.environ.get('WEBUI_THREAD_POOL_WORKERS')
            if tp_env is not None:
                tpw = int(tp_env)
        tb_pfx = parsed.tensorboard_path_prefix or os.environ.get('WEBUI_TENSORBOARD_PREFIX')
        args = WebUIArguments(
            server_name=parsed.server_name,
            server_port=parsed.server_port,
            lang=parsed.lang,
            thread_pool_workers=tpw,
            tensorboard_path_prefix=tb_pfx or '/tensorboard',
        )

    server = os.environ.get('WEBUI_SERVER') or args.server_name
    port_env = os.environ.get('WEBUI_PORT')
    port = int(port_env) if port_env else getattr(args, 'server_port', 7861)

    thread_pool_workers = getattr(args, 'thread_pool_workers', None)
    if thread_pool_workers is None:
        tp_env = os.environ.get('WEBUI_THREAD_POOL_WORKERS')
        if tp_env is not None:
            thread_pool_workers = int(tp_env)
    effective_threads = (thread_pool_workers if thread_pool_workers is not None
                         else _default_thread_pool_workers())
    logger.info(
        f'Starting SWIFT Web-UI Restful on http://{server}:{port} (thread_pool_workers={effective_threads})')
    application = create_app(
        thread_pool_workers=thread_pool_workers,
        tensorboard_path_prefix=getattr(args, 'tensorboard_path_prefix', None),
    )
    uvicorn.run(application, host=server, port=port)
