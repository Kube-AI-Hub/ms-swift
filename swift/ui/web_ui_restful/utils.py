# Copyright (c) ModelScope Contributors. All rights reserved.
import collections
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psutil
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from swift.utils import format_time, get_device_count, get_logger

logger = get_logger()

MAX_LOG_LINES = int(os.environ.get('MAX_LOG_LINES', 200))


def _normalize_gpu_value_for_ui(gpu_val: str) -> str:
    if gpu_val == '0' and get_device_count() == 1:
        if is_torch_npu_available():
            return 'npu'
        if is_torch_cuda_available():
            return 'gpu'
    return gpu_val


def get_running_tasks(cmd_name: str) -> List[Dict]:
    """Return list of running swift tasks matching the given subcommand."""
    process_name = 'swift'
    negative_name = 'swift.exe'
    tasks = []
    for proc in psutil.process_iter():
        try:
            cmdlines = proc.cmdline()
        except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if (any(process_name in c for c in cmdlines)
                and not any(negative_name in c for c in cmdlines)
                and any(cmd_name == c for c in cmdlines)):
            tasks.append(_build_task_info(proc))
    return tasks


def get_all_running_tasks() -> List[Dict]:
    """Return all swift tasks (any subcommand)."""
    cmds = ['sft', 'pt', 'rlhf', 'deploy', 'export', 'eval', 'sample']
    process_name = 'swift'
    negative_name = 'swift.exe'
    tasks = []
    for proc in psutil.process_iter():
        try:
            cmdlines = proc.cmdline()
        except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if (any(process_name in c for c in cmdlines)
                and not any(negative_name in c for c in cmdlines)
                and any(c in cmds for c in cmdlines)):
            tasks.append(_build_task_info(proc))
    return tasks


def _parse_cmdline_args(cmdlines: List[str]) -> Dict[str, str]:
    """Parse --key value pairs from a process cmdline list (safe: no string join/split)."""
    multi_value_keys = {'dataset', 'reward_funcs'}
    args: Dict[str, str] = {}
    i = 0
    while i < len(cmdlines):
        tok = cmdlines[i]
        if tok.startswith('--') and i + 1 < len(cmdlines) and not cmdlines[i + 1].startswith('--'):
            key = tok[2:]
            if key in multi_value_keys:
                values: List[str] = []
                j = i + 1
                while j < len(cmdlines) and not cmdlines[j].startswith('--'):
                    values.append(cmdlines[j])
                    j += 1
                args[key] = values
                i = j
            else:
                args[key] = cmdlines[i + 1]
                i += 2
        else:
            i += 1
    return args


def _build_task_info(proc) -> Dict:
    ts = time.time()
    create_time = proc.create_time()
    create_time_formatted = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M')
    cmdlines = proc.cmdline()
    # detect swift subcommand
    cmd = ''
    for known in ['sft', 'pt', 'rlhf', 'deploy', 'export', 'eval', 'sample']:
        if known in cmdlines:
            cmd = known
            break
    log_file = _parse_log_file(cmdlines)
    parsed_args = _parse_cmdline_args(cmdlines)
    # GPU IDs are passed via env-vars, not cmdline; read them from the process env
    try:
        env = proc.environ()
        gpu_val = (env.get('CUDA_VISIBLE_DEVICES')
                   or env.get('ASCEND_RT_VISIBLE_DEVICES')
                   or env.get('MLU_VISIBLE_DEVICES'))
        if gpu_val:
            parsed_args['gpu_ids'] = _normalize_gpu_value_for_ui(gpu_val)
    except Exception:
        pass
    return {
        'pid': proc.pid,
        'cmd': cmd,
        'create_time': create_time_formatted,
        'running': format_time(ts - create_time),
        'log_file': log_file,
        'cmdline': ' '.join(cmdlines),
        'args': parsed_args,
    }


def _parse_log_file(cmdlines: List[str]) -> Optional[str]:
    """Extract --log_file value from cmdline list."""
    for i, c in enumerate(cmdlines):
        if c == '--log_file' and i + 1 < len(cmdlines):
            return cmdlines[i + 1]
    return None


def kill_task_by_log(log_file: str) -> bool:
    """Kill a task identified by its log file path."""
    if sys.platform == 'win32':
        # find pid via psutil
        for proc in psutil.process_iter():
            try:
                cmdlines = proc.cmdline()
            except Exception:
                continue
            if log_file in cmdlines:
                command = ['taskkill', '/f', '/t', '/pid', str(proc.pid)]
                result = subprocess.run(command, capture_output=True, text=True)
                return result.returncode == 0
        return False
    else:
        result = subprocess.run(['pkill', '-9', '-f', log_file], capture_output=True, text=True)
        return result.returncode == 0


def tail_log_file(log_file: str):
    """Generator that yields new lines from a log file (blocking tail)."""
    lines = collections.deque(maxlen=MAX_LOG_LINES)
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            fail_cnt = 0
            buf = ''
            while True:
                chunk = f.read(4096)
                if not chunk:
                    time.sleep(0.3)
                    fail_cnt += 1
                    if fail_cnt > 100:
                        break
                    continue
                fail_cnt = 0
                buf += chunk
                if '\n' not in buf:
                    continue
                parts = buf.split('\n')
                if buf[-1] != '\n':
                    buf = parts[-1]
                    parts = parts[:-1]
                else:
                    buf = ''
                lines.extend(parts)
                yield '\n'.join(lines)
    except IOError:
        pass


def get_used_ports(cmd_name: str = 'deploy') -> set:
    ports = set()
    for proc in psutil.process_iter():
        try:
            cmdlines = proc.cmdline()
        except Exception:
            continue
        if 'swift' in ' '.join(cmdlines) and cmd_name in cmdlines:
            lf = _parse_log_file(cmdlines)
            if lf:
                # try to read port from cmdline
                for i, c in enumerate(cmdlines):
                    if c == '--port' and i + 1 < len(cmdlines):
                        try:
                            ports.add(int(cmdlines[i + 1]))
                        except ValueError:
                            pass
    return ports
