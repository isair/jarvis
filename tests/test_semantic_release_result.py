"""Release preparation distinguishes no changes from failed publication."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(('output', 'exit_code', 'version'), [
    ('There are no relevant changes', 0, None),
    ('Published release 1.2.3 on default channel', 0, '1.2.3'),
    ('GitHub authentication failed', 1, None),
    ('Published release 1.2.3 on default channel\nNetwork failure', 7, None),
])
def test_release_preparation_reports_its_actual_result(tmp_path, output, exit_code, version):
    workflow = yaml.safe_load(Path('.github/workflows/release.yml').read_text())
    step = next(item for item in workflow['jobs']['semantic-release']['steps']
                if item.get('id') == 'semantic')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    stub = binaries / 'npx'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$OWNED_RELEASE_OUTPUT"\nexit "$OWNED_RELEASE_EXIT"\n')
    stub.chmod(0o755)
    outputs = tmp_path / 'outputs'
    environment = {
        'PATH': str(binaries) + os.pathsep + os.defpath,
        'GITHUB_OUTPUT': str(outputs),
        'OWNED_RELEASE_OUTPUT': output,
        'OWNED_RELEASE_EXIT': str(exit_code),
    }
    result = subprocess.run([shutil.which('bash'), '-eo', 'pipefail', '-c', step['run']],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=10)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert output in result.stdout, 'Preparation diagnostics must remain visible'
    published = outputs.read_text() if outputs.exists() else ''
    if exit_code:
        assert not published, 'Failed preparation must not report release outputs'
    elif version:
        assert f'new_release_version={version}\n' in published
        assert f'new_release_git_tag=v{version}\n' in published
        assert 'new_release_published=true\n' in published
    else:
        assert published == 'new_release_published=false\n'
