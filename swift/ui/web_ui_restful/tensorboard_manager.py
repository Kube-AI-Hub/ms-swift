# Copyright (c) ModelScope Contributors. All rights reserved.
"""Start a local TensorBoard subprocess and proxy HTTP to it under a path prefix."""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()

_tb_lock = threading.Lock()
_state: Dict[str, Any] = {
    'proc': None,
    'port': None,
    'logging_dir': None,
    'path_prefix': '/tensorboard',
}


def normalize_path_prefix(prefix: str) -> str:
    p = (prefix or '/tensorboard').strip()
    if not p.startswith('/'):
        p = '/' + p
    return p.rstrip('/') or '/'


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _wait_tensorboard_ready(port: int, path_prefix: str, timeout: float = 15.0) -> bool:
    base = f'http://127.0.0.1:{port}{path_prefix}/'
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base, timeout=2.0)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def stop_tensorboard() -> None:
    with _tb_lock:
        proc = _state.get('proc')
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                logger.warning(f'TensorBoard stop: {e}')
        _state['proc'] = None
        _state['port'] = None
        _state['logging_dir'] = None


def start_tensorboard(logging_dir: str, path_prefix: str) -> Tuple[bool, str, Optional[int], Optional[str]]:
    """
    Start TensorBoard for logging_dir. Reuses running instance if same logging_dir.

    Returns:
        (ok, message, port, public_url_suffix) — public_url_suffix e.g. '/tensorboard/'
    """
    from transformers import is_tensorboard_available

    if not is_tensorboard_available():
        return False, 'tensorboard_not_installed', None, None

    _s = logging_dir.strip()
    abs_dir = os.path.abspath(_s) if not os.path.isabs(_s) else _s
    if not abs_dir or not os.path.isdir(abs_dir):
        return False, 'logging_dir_not_found', None, None

    prefix_for_tb = normalize_path_prefix(path_prefix)

    with _tb_lock:
        if (_state.get('proc') is not None and _state['proc'].poll() is None
                and _state.get('logging_dir') == abs_dir and _state.get('path_prefix') == prefix_for_tb):
            port = _state['port']
            return True, 'already_running', port, f'{prefix_for_tb}/'

        proc_old = _state.get('proc')
        if proc_old is not None:
            try:
                proc_old.terminate()
                try:
                    proc_old.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc_old.kill()
            except Exception:
                pass
            _state['proc'] = None

    port = _pick_free_port()
    cmd = [
        'tensorboard',
        '--logdir',
        abs_dir,
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        '--path_prefix',
        prefix_for_tb,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, 'tensorboard_command_not_found', None, None
    except Exception as e:
        return False, str(e), None, None

    if not _wait_tensorboard_ready(port, prefix_for_tb):
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return False, 'tensorboard_start_timeout', None, None

    with _tb_lock:
        _state['proc'] = proc
        _state['port'] = port
        _state['logging_dir'] = abs_dir
        _state['path_prefix'] = prefix_for_tb

    return True, 'started', port, f'{prefix_for_tb}/'


def get_tensorboard_backend_origin() -> Optional[str]:
    """http://127.0.0.1:<port> when the TensorBoard subprocess is running."""
    with _tb_lock:
        port = _state.get('port')
        proc = _state.get('proc')
        if port is None or proc is None or proc.poll() is not None:
            return None
        return f'http://127.0.0.1:{port}'


def get_tensorboard_upstream_base() -> Optional[str]:
    with _tb_lock:
        port = _state.get('port')
        pfx = _state.get('path_prefix')
        proc = _state.get('proc')
        if port is None or not pfx or proc is None or proc.poll() is not None:
            return None
        return f'http://127.0.0.1:{port}{pfx}/'


def tensorboard_status() -> Dict[str, Any]:
    with _tb_lock:
        proc = _state.get('proc')
        if proc is None or proc.poll() is not None:
            return {'running': False, 'logging_dir': None, 'port': None}
        return {
            'running': True,
            'logging_dir': _state.get('logging_dir'),
            'port': _state.get('port'),
            'path_prefix': _state.get('path_prefix'),
        }


# Hop-by-hop and proxy-specific headers to drop when forwarding
_HOP_HEADERS = frozenset({
    'host', 'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailers', 'transfer-encoding', 'upgrade', 'content-length',
})


def filter_request_headers(headers: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in _HOP_HEADERS:
            continue
        out[k] = v
    return out


def filter_response_headers(headers: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in _HOP_HEADERS:
            continue
        out[k] = v
    return out
