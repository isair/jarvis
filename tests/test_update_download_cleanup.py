"""Downloaded archives are disposable after the installer stages their payload."""

from types import SimpleNamespace
from unittest.mock import patch
import pytest
from desktop_app import update_dialog

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


@pytest.mark.parametrize('success', [True, False])
def test_download_archive_removed_after_install_handoff(tmp_path, success, qapp):
    archive_dir = tmp_path / 'download'
    archive_dir.mkdir()
    archive = archive_dir / 'bundle.zip'
    archive.write_bytes(b'owned downloaded archive')
    extracted = tmp_path / 'installer-stage'
    extracted.mkdir()
    payload = extracted / 'payload'
    payload.write_bytes(b'owned extracted payload')

    def install(path):
        assert path.read_bytes() == b'owned downloaded archive'
        return success

    dialog = update_dialog.UpdateProgressDialog(SimpleNamespace(asset_id=1))
    dialog._temp_dir = archive_dir
    with (
        patch.object(update_dialog, 'install_update', side_effect=install),
        patch.object(update_dialog, 'save_installed_asset_id'),
        patch.object(update_dialog.QTimer, 'singleShot'),
    ):
        dialog._install(archive)
    assert not archive_dir.exists(), 'Downloaded archive remains after staging handoff'
    assert payload.read_bytes() == b'owned extracted payload'
    dialog.deleteLater()
