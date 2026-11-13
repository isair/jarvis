"""Include the local speech detector weights in desktop builds."""
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("faster_whisper", includes=["assets/*"])
