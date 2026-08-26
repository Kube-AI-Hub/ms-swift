from pathlib import Path

from swift.dataset.loader import DatasetLoader, _is_registered_dataset_id
from swift.dataset.register import DATASET_MAPPING
from swift.hub import CSGHub


def test_csghub_jsonl_dataset_ignores_incomplete_dataset_infos(monkeypatch, tmp_path):
    load_calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        local_dir = Path(kwargs['local_dir'])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / 'self_cognition.jsonl').write_text(
            '{"query": "Who are you?", "response": "Assistant", "tag": "identity"}\n',
            encoding='utf-8',
        )
        (local_dir / 'dataset_infos.json').write_text(
            '{"default":{"features":{"query":{"_type":"Value"}}}}',
            encoding='utf-8',
        )
        return str(local_dir)

    def fake_load_dataset(path, **kwargs):
        load_calls.append((path, kwargs))
        return 'dataset'

    monkeypatch.setattr(CSGHub, '_lookup_default_branch', classmethod(lambda cls, *args, **kwargs: None))
    monkeypatch.setattr('pycsghub.snapshot_download.snapshot_download', fake_snapshot_download)
    monkeypatch.setattr('datasets.load_dataset', fake_load_dataset)

    result = CSGHub.load_dataset(
        'admin/self-cognition',
        None,
        'train',
        revision=None,
        cache_dir=str(tmp_path),
        num_proc=2,
    )

    assert result == 'dataset'
    assert load_calls[0][0] == 'json'
    assert load_calls[0][1]['data_files'] == [
        str(tmp_path / 'snapshots' / 'dataset' / 'admin--self-cognition' / 'main' / 'self_cognition.jsonl')
    ]
    assert load_calls[0][1]['num_proc'] == 2


def test_registered_dataset_id_detection():
    meta = DATASET_MAPPING['self-cognition']
    assert _is_registered_dataset_id('self-cognition', meta)
    assert _is_registered_dataset_id('swift/self-cognition', meta)
    assert _is_registered_dataset_id('modelscope/self-cognition', meta)
    assert not _is_registered_dataset_id('admin/self-cognition', meta)


def test_csghub_revision_candidates_prefer_repo_default_branch(monkeypatch):
    monkeypatch.setattr(CSGHub, '_lookup_default_branch', classmethod(lambda cls, *args, **kwargs: 'master'))

    assert CSGHub._revision_candidates(None, 'admin/demo', 'model', None, 'http://csghub') == [
        'master',
        'main',
    ]
    assert CSGHub._revision_candidates('master', 'admin/demo', 'model', None, 'http://csghub') == [
        'master',
        'main',
    ]
    assert CSGHub._revision_candidates('main', 'admin/demo', 'model', None, 'http://csghub') == [
        'main',
        'master',
    ]
    assert CSGHub._revision_candidates('v1.0', 'admin/demo', 'model', None, 'http://csghub') == ['v1.0']


def test_csghub_download_model_retries_master_when_main_has_no_sha(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append(kwargs['revision'])
        if kwargs['revision'] == 'main':
            raise AssertionError('Repo info returned from server must have a revision sha.')
        return str(tmp_path / kwargs['revision'])

    monkeypatch.setattr(CSGHub, '_lookup_default_branch', classmethod(lambda cls, *args, **kwargs: None))
    monkeypatch.setattr('pycsghub.snapshot_download.snapshot_download', fake_snapshot_download)

    result = CSGHub.download_model('admin/DeepSeek-R1-Distill-Qwen-1.5B', revision=None, cache_dir=str(tmp_path))

    assert calls == ['main', 'master']
    assert result == str(tmp_path / 'master')


def test_csghub_repo_id_not_remapped_to_registry(monkeypatch):
    cascade_calls = []

    def fake_cascade(**kwargs):
        cascade_calls.append(kwargs)
        return 'dataset'

    monkeypatch.setattr('swift.dataset.loader.cascading_load_dataset', fake_cascade)

    loader = DatasetLoader()
    meta = DATASET_MAPPING['self-cognition']
    loader._load_repo_dataset(
        'admin/self-cognition',
        meta.subsets[0],
        use_hf=None,
        dataset_meta=meta,
    )

    assert cascade_calls[0]['csg_dataset_id'] == 'admin/self-cognition'
    assert cascade_calls[0]['hf_dataset_id'] == 'admin/self-cognition'
    assert cascade_calls[0]['ms_dataset_id'] == 'admin/self-cognition'
