from pathlib import Path
import json

from fastapi.testclient import TestClient

from swift.ui.web_ui_restful.app import (
    FRONTEND_DIR,
    _fallback_device_type,
    create_app,
    latest_checkpoint_dir,
    resolve_export_source,
)


def test_frontend_has_local_chartjs():
    chart_js = FRONTEND_DIR / 'chart.umd.min.js'
    index_html = (FRONTEND_DIR / 'index.html').read_text(encoding='utf-8')

    assert chart_js.is_file(), 'chart.umd.min.js must be vendored for offline deployments'
    assert chart_js.stat().st_size > 100_000
    assert 'cdn.jsdelivr.net' not in index_html
    assert './chart.umd.min.js' in index_html


def test_frontend_serves_vendored_chartjs():
    client = TestClient(create_app())
    response = client.get('/chart.umd.min.js')

    assert response.status_code == 200
    assert 'Chart' in response.text


def test_fallback_device_type():
    assert _fallback_device_type(['npu', 'cpu']) == 'npu'
    assert _fallback_device_type(['mlu', 'cpu']) == 'mlu'
    assert _fallback_device_type(['mps', 'cpu']) == 'mps'
    assert _fallback_device_type(['0', '1', 'cpu']) == 'gpu'
    assert _fallback_device_type(['cpu']) == 'cpu'


def test_frontend_export_csghub_fields():
    index_html = (FRONTEND_DIR / 'index.html').read_text(encoding='utf-8')
    app_js = (FRONTEND_DIR / 'app.js').read_text(encoding='utf-8')
    assert 'id="export-push-to-hub"' in index_html
    assert 'id="export-hub-model-id"' in index_html
    assert 'id="export-adapters"' in index_html
    assert 'export-push-to-hub' in app_js
    assert 'push_to_hub:' in app_js
    assert 'function prefillExportFromTrain' in app_js
    assert '/api/v1/export/from-train' in app_js


def test_latest_checkpoint_dir_prefers_highest_step(tmp_path: Path):
    run = tmp_path / 'output' / 'qwen-run'
    older = run / 'checkpoint-10'
    newer = run / 'checkpoint-80'
    older.mkdir(parents=True)
    newer.mkdir()
    (older / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (newer / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (run / 'args.json').write_text('{}', encoding='utf-8')

    assert latest_checkpoint_dir(str(run)).endswith('checkpoint-80')
    assert latest_checkpoint_dir(str(newer)).endswith('checkpoint-80')
    assert latest_checkpoint_dir(str(tmp_path / 'missing')) is None


def test_export_from_train_fills_merge_paths(tmp_path: Path):
    run = tmp_path / 'output' / 'qwen-run'
    ckpt = run / 'checkpoint-40'
    ckpt.mkdir(parents=True)
    (ckpt / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (ckpt / 'args.json').write_text(
        json.dumps({
            'model': 'Qwen/Qwen3-0.6B',
            'model_type': 'qwen3',
            'template': 'qwen3',
            'tuner_type': 'lora',
        }),
        encoding='utf-8',
    )

    source = resolve_export_source(output_dir=str(run), tuner_type='lora')
    assert source['found'] is True
    assert source['adapters'].endswith('checkpoint-40')
    assert source['output_dir'].endswith('checkpoint-40-merged')
    assert source['merge_lora'] is True
    assert source['model'] == 'Qwen/Qwen3-0.6B'
    assert source['template'] == 'qwen3'

    with TestClient(create_app()) as client:
        data = client.get('/api/v1/export/from-train', params={'output_dir': str(run)}).json()
    assert data['adapters'].endswith('checkpoint-40')
    assert data['merge_lora'] is True


def test_export_from_train_full_ft_uses_checkpoint_as_model(tmp_path: Path):
    run = tmp_path / 'output' / 'full-run'
    ckpt = run / 'checkpoint-8'
    ckpt.mkdir(parents=True)
    (ckpt / 'model.safetensors').write_bytes(b'x')
    source = resolve_export_source(output_dir=str(run), tuner_type='full')
    assert source['found'] is True
    assert source['adapters'] is None
    assert source['merge_lora'] is False
    assert source['model'].endswith('checkpoint-8')


def test_export_from_train_marks_merge_before_checkpoint(tmp_path: Path):
    run = tmp_path / 'output' / 'pending'
    run.mkdir(parents=True)
    (run / 'args.json').write_text(
        json.dumps({'model': 'Qwen/Qwen3-0.6B', 'tuner_type': 'lora'}),
        encoding='utf-8',
    )
    source = resolve_export_source(output_dir=str(run))
    assert source['found'] is True
    assert source['adapters'] is None
    assert source['merge_lora'] is True
    assert source['model'] == 'Qwen/Qwen3-0.6B'

