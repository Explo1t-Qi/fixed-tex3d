"""Exercise real Bash audit functions without touching server assets or Git."""

import os
from pathlib import Path
import re
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/step1b_server_run.sh"


@pytest.mark.parametrize("head,dirty,backup,find_error,message", [
    ("different", "", "", "0", "HEAD mismatch: expected=expected actual=different"),
    ("expected", " M scripts/step1b_server_run.sh", "", "0", "dirty worktree"),
    ("expected", "", "/assets/old_clean_backup.xml", "0", "backup files remain"),
    ("expected", "", "", "1", "backup search failed"),
    ("expected", "", "", "0", "AUDIT PASSED"),
])
def test_audit_identifies_failed_predicate(head, dirty, backup, find_error, message):
    definitions = re.findall(
        r"^(?:git_audit|asset_and_git_audit)\(\) \{.*?^\}",
        SCRIPT.read_text(), flags=re.MULTILINE | re.DOTALL,
    )
    harness = r'''
set -euo pipefail
EXPECTED_HEAD=expected
ASSET_DIR=/assets
LIBERO_ROOT_PATH=/libero
git() {
    case "$*" in
        'rev-parse HEAD') printf '%s\n' "$MOCK_HEAD" ;;
        'status --porcelain') printf '%s' "$MOCK_DIRTY" ;;
        'status --short --branch') printf '## test\n%s\n' "$MOCK_DIRTY" ;;
        *) return 2 ;;
    esac
}
sha256sum() { while read -r line; do :; done; printf 'XML: OK\ntexture: OK\n'; }
find() { printf '%s' "$MOCK_BACKUP"; return "$MOCK_FIND_ERROR"; }
'''
    result = subprocess.run(
        ["bash"], input=harness + "\n".join(definitions)
        + '\nasset_and_git_audit\nprintf "AUDIT PASSED\\n"\n',
        text=True, capture_output=True,
        env={**os.environ, "MOCK_HEAD": head, "MOCK_DIRTY": dirty,
             "MOCK_BACKUP": backup, "MOCK_FIND_ERROR": find_error},
    )
    assert message in result.stdout + result.stderr
    assert result.returncode == (0 if message == "AUDIT PASSED" else 1)
    if dirty:
        assert dirty in result.stdout + result.stderr


@pytest.mark.parametrize("existing_run", [False, True])
def test_server_preflight_uses_gpu1_and_never_starts_training(tmp_path, existing_run):
    source = SCRIPT.read_text()
    directories = ("REPO", "LOG_ROOT", "OPENVLA_CKPT", "LIBERO_ROOT_PATH",
                   "OPENPI_ROOT", "SHARED_ROOT", "PI05_CKPT")
    for key in (*directories, "OPENVLA_PY", "JOINT_PY"):
        path = tmp_path / key
        if key in directories:
            path.mkdir()
        else:
            path.write_text("#!/usr/bin/env bash\necho UNEXPECTED_PYTHON_CALL\nexit 99\n")
            path.chmod(0o755)
        source = re.sub(rf"^export {key}=.*$", f"export {key}='{path}'",
                        source, flags=re.MULTILINE)
    (tmp_path / "PI05_CKPT/model.safetensors").touch()
    old_log = tmp_path / "LOG_ROOT/step1b-mature-o2-5000-v1.console.log"
    old_log.write_text("previous failed preflight\n")
    run = tmp_path / "LOG_ROOT/step1b-mature-o2-5000-v1"
    if existing_run:
        run.mkdir()
    mocks = r'''
git() {
    case "$*" in
        'rev-parse HEAD'|'rev-parse refs/heads/feat/step1b-mature-o2-trajectory') printf 'expected\n' ;;
        'status --porcelain'|'switch feat/step1b-mature-o2-trajectory') return 0 ;;
        'status --short --branch') printf '## test\n' ;;
        *) return 2 ;;
    esac
}
sha256sum() { while read -r line; do :; done; printf 'XML: OK\ntexture: OK\n'; }
find() { return 0; }
'''
    result = subprocess.run(["bash", "-s", "--", "expected", "--preflight-only"],
                            input=mocks + source, text=True, capture_output=True)
    assert "GPU=1" in result.stdout
    assert "UNEXPECTED_PYTHON_CALL" not in result.stdout + result.stderr
    assert result.returncode == (1 if existing_run else 0)
    assert ("run directory already exists" if existing_run else "PREFLIGHT PASSED") in result.stdout + result.stderr
    assert old_log.read_text() == "previous failed preflight\n"
    assert sorted(p.name for p in old_log.parent.glob("*.log")) == [old_log.name]
    assert run.exists() is existing_run
