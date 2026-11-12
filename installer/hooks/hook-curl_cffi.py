"""Bundle the search client's native libraries and fingerprint data."""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('curl_cffi')
