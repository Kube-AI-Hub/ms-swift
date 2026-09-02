from pathlib import Path
import json

from fastapi.testclient import TestClient

from swift.ui.web_ui_restful.app import (
    FRONTEND_DIR,
    TrainRequest,
    _build_train_command,
    _fallback_device_type,
    create_app,
    latest_checkpoint_dir,
    resolve_export_source,
    resolve_train_resume_hint,
    trainer_state_unfinished,
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
    assert 'checked' in index_html.split('id="export-push-to-hub"', 1)[1].split('>', 1)[0]
    assert 'id="export-hub-model-id"' in index_html
    assert 'id="export-adapters"' in index_html
    assert 'export-push-to-hub' in app_js
    assert 'push_to_hub:' in app_js
    assert 'function prefillExportFromTrain' in app_js
    assert '/api/v1/export/from-train' in app_js
    assert 'function startExportWatch' in app_js
    assert 'function restoreExportPage' in app_js
    assert 'swift.export.last' in app_js
    assert 'applyExportPushSideEffects' in app_js


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


def _write_swift_ckpt(run: Path, step: int, global_step: int, max_steps: int, train_loss=None, scheduler=False):
    ckpt = run / f'checkpoint-{step}'
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / 'adapter_config.json').write_text('{}', encoding='utf-8')
    state = {'global_step': global_step, 'max_steps': max_steps, 'log_history': []}
    if train_loss is not None:
        state['log_history'].append({'step': global_step, 'train_loss': train_loss})
    (ckpt / 'trainer_state.json').write_text(json.dumps(state), encoding='utf-8')
    if scheduler:
        (ckpt / 'scheduler.pt').write_bytes(b'x')
    return ckpt


def test_trainer_state_unfinished_uses_steps_and_train_loss():
    unfinished, gs, ms = trainer_state_unfinished({'global_step': 40, 'max_steps': 246, 'log_history': []})
    assert unfinished is True
    assert gs == 40
    assert ms == 246

    finished, gs, ms = trainer_state_unfinished({
        'global_step': 246,
        'max_steps': 246,
        'log_history': [{'train_loss': 0.07}],
    })
    assert finished is False
    assert gs == 246


def test_resume_hint_finds_unfinished_checkpoint(tmp_path: Path):
    run = tmp_path / 'output' / 'qwen-run'
    ckpt = _write_swift_ckpt(run, 40, global_step=40, max_steps=246)
    hint = resolve_train_resume_hint(output_dir=str(run))
    assert hint['found'] is True
    assert hint['checkpoint'].endswith('checkpoint-40')
    assert hint['global_step'] == 40
    assert hint['max_steps'] == 246
    assert hint['resume_only_model'] is True
    assert str(ckpt.parent.resolve()) == hint['output_dir']


def test_resume_hint_ignores_finished_run(tmp_path: Path):
    run = tmp_path / 'output' / 'done-run'
    _write_swift_ckpt(run, 246, global_step=246, max_steps=246, train_loss=0.06, scheduler=True)
    hint = resolve_train_resume_hint(output_dir=str(run))
    assert hint['found'] is False


def test_resume_hint_api_and_train_command_resume_flags(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('swift.ui.web_ui_restful.app.get_running_tasks', lambda cmd: [])
    run = tmp_path / 'output' / 'qwen-run'
    ckpt = _write_swift_ckpt(run, 80, global_step=80, max_steps=246, scheduler=True)

    with TestClient(create_app()) as client:
        missing = client.get('/api/v1/train/resume-hint', params={'model': 'Qwen/Qwen3-0.6B'})
        found = client.get(
            '/api/v1/train/resume-hint',
            params={'model': 'Qwen/Qwen3-0.6B', 'output_dir': str(run)},
        )
        done_dir = tmp_path / 'output' / 'done'
        _write_swift_ckpt(done_dir, 246, global_step=246, max_steps=246, train_loss=0.1)
        finished = client.get(
            '/api/v1/train/resume-hint',
            params={'model': 'Qwen/Qwen3-0.6B', 'output_dir': str(done_dir)},
        )

    assert missing.status_code == 200
    assert missing.json()['found'] is False
    assert found.status_code == 200
    payload = found.json()
    assert payload['found'] is True
    assert payload['checkpoint'].endswith('checkpoint-80')
    assert payload['resume_only_model'] is False
    assert finished.json()['found'] is False

    monkeypatch.setattr('swift.ui.web_ui_restful.app.get_running_tasks', lambda cmd: [{'pid': 1}])
    with TestClient(create_app()) as client:
        blocked = client.get(
            '/api/v1/train/resume-hint',
            params={'model': 'Qwen/Qwen3-0.6B', 'output_dir': str(run)},
        )
    assert blocked.json()['found'] is False

    req = TrainRequest(
        model='Qwen/Qwen3-0.6B',
        resume_from_checkpoint=str(ckpt),
        output_dir=str(run),
        dry_run=True,
    )
    command, _, _, _ = _build_train_command(req, 'sft')
    assert '--resume_from_checkpoint' in command
    resume_val = command[command.index('--resume_from_checkpoint') + 1]
    assert resume_val.endswith('checkpoint-80')
    assert '--resume_only_model' not in command

    req_partial = TrainRequest(
        model='Qwen/Qwen3-0.6B',
        resume_from_checkpoint=str(_write_swift_ckpt(tmp_path / 'output' / 'partial', 10, 10, 100)),
        dry_run=True,
    )
    command_partial, _, _, logging_dir = _build_train_command(req_partial, 'sft')
    assert command_partial[command_partial.index('--resume_only_model') + 1] == 'true'
    assert '--output_dir' in command_partial
    assert logging_dir.endswith('partial') or 'partial' in command_partial[command_partial.index('--output_dir') + 1]


def test_frontend_resume_hint_wiring():
    app_js = (FRONTEND_DIR / 'app.js').read_text(encoding='utf-8')
    i18n = (FRONTEND_DIR / 'index-i18n.js').read_text(encoding='utf-8')
    assert '/api/v1/train/resume-hint' in app_js
    assert 'resume_from_checkpoint' in app_js
    assert "btnContinue: '继续'" in i18n
    assert "btnContinue: 'Continue'" in i18n
    assert 'function refreshTrainResumeHint' in app_js
    assert "key = useContinue ? 'btnContinue' : 'btnStart'" in app_js

