from pathlib import Path

from swift.model.utils import copy_base_model_readme, save_checkpoint


QWEN_README = """---
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
---

# Qwen3-0.6B
"""

ADAPTER_README = """## Training procedure

- SWIFT 4.5.2
"""


def test_copy_base_model_readme_keeps_pipeline_tag(tmp_path: Path):
    base = tmp_path / 'base'
    export_dir = tmp_path / 'export'
    base.mkdir()
    (base / 'README.md').write_text(QWEN_README, encoding='utf-8')

    assert copy_base_model_readme([str(base)], str(export_dir))
    copied = (export_dir / 'README.md').read_text(encoding='utf-8')
    assert 'pipeline_tag: text-generation' in copied
    assert copied == QWEN_README


def test_copy_base_model_readme_overwrites_adapter_card(tmp_path: Path):
    base = tmp_path / 'base'
    export_dir = tmp_path / 'export'
    base.mkdir()
    export_dir.mkdir()
    (base / 'README.md').write_text(QWEN_README, encoding='utf-8')
    (export_dir / 'README.md').write_text(ADAPTER_README, encoding='utf-8')

    assert copy_base_model_readme([str(base)], str(export_dir))
    assert (export_dir / 'README.md').read_text(encoding='utf-8') == QWEN_README


def test_copy_base_model_readme_missing_src(tmp_path: Path):
    assert copy_base_model_readme([str(tmp_path / 'missing')], str(tmp_path / 'export')) is False
    assert not (tmp_path / 'export' / 'README.md').exists()


def test_save_checkpoint_prefers_base_readme_over_adapter(tmp_path: Path):
    base = tmp_path / 'base'
    adapter = tmp_path / 'adapter'
    export_dir = tmp_path / 'export'
    base.mkdir()
    adapter.mkdir()
    (base / 'README.md').write_text(QWEN_README, encoding='utf-8')
    (adapter / 'README.md').write_text(ADAPTER_README, encoding='utf-8')

    class DummyModel:
        model_dir = str(base)

        def save_pretrained(self, *args, **kwargs):
            return None

    class DummyProcessor:
        def save_pretrained(self, *args, **kwargs):
            return None

    save_checkpoint(
        DummyModel(),
        DummyProcessor(),
        str(export_dir),
        model_dirs=[str(adapter)],
    )
    assert (export_dir / 'README.md').read_text(encoding='utf-8') == QWEN_README
    # Adapter files listed in additional_saved_files still copy; README must not
    # stay as the training card.
    assert 'pipeline_tag: text-generation' in (export_dir / 'README.md').read_text(encoding='utf-8')
