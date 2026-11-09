"""Release assembly, the source manifest and replay keys; no model calls."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts.prepare_experiments import ROOT, METHOD, prepare
from scripts.build_replay import build


def test_source_manifest_matches_relocated_files():
    manifest = json.loads((ROOT / 'experiments/provenance.json').read_text())
    assert manifest['files']
    for item in manifest['files']:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
        assert len(item['source_sha256']) == 64


def test_replay_keeps_duplicate_order_and_dates(tmp_path):
    trace = tmp_path / 'one.json'
    trace.write_text(json.dumps({'steps': [
        {'query': 'q', 'qa_metadata': {'question_date': date}, 'retrieved_memory': {'value': value}}
        for date, value in [('2023', 1), ('2023', 2), ('2024', 3)]
    ]}))
    replay = build(tmp_path, 'longmemeval', baseline='mem0')
    assert replay['entries'][json.dumps(['q', '2023'])] == [{'value': 1}, {'value': 2}]
    assert replay['entries'][json.dumps(['q', '2024'])] == [{'value': 3}]
    assert replay['baseline'] == 'mem0'
    trace.write_text(json.dumps({'steps': [{'query': 'q'}]}))
    with pytest.raises(ValueError, match='Missing recorded retrieval'):
        build(tmp_path, 'longmemeval')


def test_locomo_replay_uses_speaker_identity(tmp_path):
    traces = tmp_path / 'traces'; traces.mkdir()
    (traces / 'one.json').write_text(json.dumps({'user_id': 'c1', 'steps': [
        {'query': 'q', 'retrieved_memory': {'notes': ['n']}}
    ]}))
    data = tmp_path / 'data.json'
    data.write_text(json.dumps([{'sample_id': 'c1', 'conversation': {'speaker_a': 'A', 'speaker_b': 'B'}}]))
    replay = build(traces, 'locomo', locomo_data=data)
    assert json.dumps(['A', 'B', 'q']) in replay['entries']


def test_prepared_experiments_resolve_configs_and_import_in_isolation():
    from evolution.heldout import _heldout_arg_parser
    from evolution.orchestrator import _resolve_config, HELDOUT_REQUIRED_SCHEMA
    workspace = ROOT / 'workspace'; workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='release-test-', dir=workspace) as td:
        dest = Path(td) / 'experiments'
        configs = prepare(dest)
        assert len(configs) == 20
        with pytest.raises(FileExistsError):
            prepare(dest)
        for path in configs:
            args = _heldout_arg_parser().parse_args(['--config', str(path)])
            cfg = _resolve_config(args, required_schema=HELDOUT_REQUIRED_SCHEMA)
            harness = ROOT / cfg['harnesses'][0]
            # A fresh interpreter prevents shared src/ names between harnesses masking missing files.
            snippet = ('from pathlib import Path; from evolution.launch import _load_harness_class; '
                       'from core.memo_class import MemoClass; '
                       f'cls = _load_harness_class(Path({str(harness)!r})); '
                       'assert issubclass(cls, MemoClass); '
                       + ('' if cfg['contract']['compose'] else
                          'assert cls.compose_memory_context is MemoClass.compose_memory_context; '))
            result = subprocess.run([sys.executable, '-c', snippet], cwd=ROOT, text=True, capture_output=True)
            assert result.returncode == 0, f'{path.name} in {path.parent.name}: {result.stderr}'
            for name in ('mape_a.py', 'memo_lme.py'):
                if (harness / name).exists():
                    assert (harness / name).read_bytes() == (METHOD / 'memo.py').read_bytes()
