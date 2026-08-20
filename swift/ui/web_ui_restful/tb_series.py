# Copyright (c) ModelScope Contributors. All rights reserved.
"""TensorBoard scalar series for SFT/PT training charts (aligned with Gradio Runtime.sft_plot)."""
import os
from typing import Any, Dict, List, Optional

from swift.utils import read_tensorboard_file, tensorboard_smoothing

# Same order and smoothing as swift.ui.llm_train.runtime.Runtime.sft_plot
SFT_PLOT: List[Dict[str, Any]] = [
    {'name': 'train/loss', 'smooth': 0.9},
    {'name': 'train/acc', 'smooth': None},
    {'name': 'train/learning_rate', 'smooth': None},
    {'name': 'eval/loss', 'smooth': 0.9},
    {'name': 'eval/acc', 'smooth': None},
]

# Aligned with Gradio Runtime.dpo_plot / kto_plot / orpo_plot / grpo_plot
DPO_PLOT: List[Dict[str, Any]] = [
    {'name': 'train/loss', 'smooth': 0.9},
    {'name': 'train/rewards/accuracies', 'smooth': None},
    {'name': 'train/rewards/margins', 'smooth': 0.9},
    {'name': 'train/logps/chosen', 'smooth': 0.9},
    {'name': 'train/logps/rejected', 'smooth': 0.9},
]

KTO_PLOT: List[Dict[str, Any]] = [
    {'name': 'kl', 'smooth': None},
    {'name': 'rewards/chosen_sum', 'smooth': 0.9},
    {'name': 'logps/chosen_sum', 'smooth': 0.9},
    {'name': 'rewards/rejected_sum', 'smooth': 0.9},
    {'name': 'logps/rejected_sum', 'smooth': 0.9},
]

ORPO_PLOT: List[Dict[str, Any]] = [
    {'name': 'train/loss', 'smooth': 0.9},
    {'name': 'train/rewards/accuracies', 'smooth': None},
    {'name': 'train/rewards/margins', 'smooth': 0.9},
    {'name': 'train/rewards/chosen', 'smooth': 0.9},
    {'name': 'train/log_odds_ratio', 'smooth': 0.9},
]

GRPO_PLOT: List[Dict[str, Any]] = [
    {'name': 'train/loss', 'smooth': 0.9},
    {'name': 'train/reward', 'smooth': 0.9},
    {'name': 'train/learning_rate', 'smooth': None},
    {'name': 'train/completions/mean_length', 'smooth': 0.9},
    {'name': 'train/kl', 'smooth': 0.9},
]

RLHF_PLOT_MAP: Dict[str, List[Dict[str, Any]]] = {
    'dpo': DPO_PLOT,
    'cpo': DPO_PLOT,
    'simpo': DPO_PLOT,
    'kto': KTO_PLOT,
    'orpo': ORPO_PLOT,
    'rm': DPO_PLOT,
    'ppo': DPO_PLOT,
    'gkd': DPO_PLOT,
}


def merge_tensorboard_scalars(logging_dir: str) -> Dict[str, List[Dict[str, float]]]:
    """
    Merge scalar tags from all events.out.tfevents.* files under logging_dir.

    TensorBoard UI reads every file in the logdir; we previously read only one file.
    Using sorted()[0] on flat filenames often picked the *oldest* shard, which could miss
    scalars written to newer files. HuggingFace may also rotate event files.
    """
    event_files: List[tuple] = []
    for root, _, files in os.walk(logging_dir):
        for f in files:
            if f.startswith('events.out'):
                fp = os.path.join(root, f)
                try:
                    mt = os.path.getmtime(fp)
                except OSError:
                    mt = 0.0
                event_files.append((mt, fp))
    event_files.sort(key=lambda x: x[0])
    by_tag_step: Dict[str, Dict[int, float]] = {}
    for _mt, fp in event_files:
        try:
            part = read_tensorboard_file(fp)
        except Exception:
            continue
        for tag, items in part.items():
            d = by_tag_step.setdefault(tag, {})
            for it in items:
                d[it['step']] = it['value']
    return {
        tag: [{'step': s, 'value': v} for s, v in sorted(d.items(), key=lambda x: x[0])]
        for tag, d in by_tag_step.items()
    }


def resolve_logging_dir_path(logging_dir: str) -> str:
    s = (logging_dir or '').strip()
    if not s:
        return ''
    return os.path.abspath(s) if not os.path.isabs(s) else s


def _pick_tb_tag_for_plot(name: str, data: Dict[str, List[Dict[str, float]]]) -> Optional[str]:
    """Map UI plot name to first matching TensorBoard tag (HF rewrite_logs / swift metrics)."""
    if name == 'train/learning_rate':
        for cand in ('train/learning_rate', 'train/lr'):
            if cand in data:
                return cand
        return None
    if name == 'train/acc':
        if 'train/token_acc' in data:
            return 'train/token_acc'
        if 'train/seq_acc' in data:
            return 'train/seq_acc'
        if 'train/accuracy' in data:
            return 'train/accuracy'
        return 'train/acc' if 'train/acc' in data else None
    if name == 'eval/loss':
        for cand in ('eval/loss', 'eval/losses'):
            if cand in data:
                return cand
        return None
    if name == 'eval/acc':
        for cand in ('eval/token_acc', 'eval/seq_acc', 'eval/accuracy', 'eval/acc'):
            if cand in data:
                return cand
        return None
    return name if name in data else None


def build_sft_pt_series(logging_dir: str) -> Dict[str, Any]:
    """Load scalar series from TensorBoard for the five SFT/PT plots."""
    abs_dir = resolve_logging_dir_path(logging_dir)
    if not abs_dir:
        return {'series': {}, 'detail': 'empty_logging_dir'}
    if not os.path.isdir(abs_dir):
        return {'series': {}, 'detail': 'logging_dir_not_found', 'logging_dir': abs_dir}

    try:
        data = merge_tensorboard_scalars(abs_dir)
    except Exception as e:
        return {'series': {}, 'detail': str(e), 'logging_dir': abs_dir}

    if not data:
        return {'series': {}, 'detail': 'no_event_file', 'logging_dir': abs_dir}

    series: Dict[str, Any] = {}
    for spec in SFT_PLOT:
        name = spec['name']
        smooth = spec['smooth']
        tb_tag = _pick_tb_tag_for_plot(name, data)
        if not tb_tag:
            continue
        _data = data[tb_tag]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        entry: Dict[str, Any] = {'tb_tag': tb_tag, 'step': steps, 'value': values}
        if smooth is not None and len(values) > 1:
            entry['value_smoothed'] = tensorboard_smoothing(values, smooth)
        series[name] = entry

    return {'series': series, 'detail': None, 'logging_dir': abs_dir}


def _pick_rlhf_tb_tag(name: str, data: Dict[str, List[Dict[str, float]]]) -> Optional[str]:
    """
    Resolve TB tag for RLHF/GRPO plot names.
    Swift's DPO trainer emits metrics as e.g. 'rewards/accuracies'; HF rewrite_logs
    adds the 'train/' prefix to produce 'train/rewards/accuracies'. We try the
    prefixed form first (canonical) then the bare form as fallback.
    """
    if name in data:
        return name
    # Try stripping the 'train/' prefix for bare-tag fallback
    if name.startswith('train/'):
        bare = name[len('train/'):]
        if bare in data:
            return bare
    # Learning-rate alias
    if name == 'train/learning_rate' and 'train/lr' in data:
        return 'train/lr'
    return None


def _build_series_from_plot(plot_config: List[Dict[str, Any]], data: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """Build series dict from a plot config list with RLHF tag fallbacks."""
    series: Dict[str, Any] = {}
    for spec in plot_config:
        name = spec['name']
        smooth = spec['smooth']
        tb_tag = _pick_rlhf_tb_tag(name, data)
        if not tb_tag:
            continue
        _data = data[tb_tag]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        entry: Dict[str, Any] = {'tb_tag': tb_tag, 'step': steps, 'value': values}
        if smooth is not None and len(values) > 1:
            entry['value_smoothed'] = tensorboard_smoothing(values, smooth)
        series[name] = entry
    return series


def _load_and_build(logging_dir: str, plot_config: List[Dict[str, Any]]) -> Dict[str, Any]:
    abs_dir = resolve_logging_dir_path(logging_dir)
    if not abs_dir:
        return {'series': {}, 'detail': 'empty_logging_dir'}
    if not os.path.isdir(abs_dir):
        return {'series': {}, 'detail': 'logging_dir_not_found', 'logging_dir': abs_dir}
    try:
        data = merge_tensorboard_scalars(abs_dir)
    except Exception as e:
        return {'series': {}, 'detail': str(e), 'logging_dir': abs_dir}
    if not data:
        return {'series': {}, 'detail': 'no_event_file', 'logging_dir': abs_dir}
    return {'series': _build_series_from_plot(plot_config, data), 'detail': None, 'logging_dir': abs_dir}


def build_rlhf_series(logging_dir: str, rlhf_type: str = 'dpo') -> Dict[str, Any]:
    plot_config = RLHF_PLOT_MAP.get(rlhf_type, DPO_PLOT)
    return _load_and_build(logging_dir, plot_config)


def build_grpo_series(logging_dir: str) -> Dict[str, Any]:
    return _load_and_build(logging_dir, GRPO_PLOT)
