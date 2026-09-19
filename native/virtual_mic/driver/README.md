# Toustovač Windows audio stack — native sources

The capture-only microphone endpoint + the user-mode broker that feeds it:

- `driver/` — WDK driver package (topology + WaveRT, no render nodes)
- `broker/` — `ToustovacAudioBroker.exe` (SCM service + named pipe server)
- `installer/` — `ToustovacAudioInstall.exe` (SetupAPI bootstrapper)
- `package/` — staged `.sys/.inf/.cat` + both `.exe`s

See the root `plan_clean_mic.md` (sections 4–12) for the authoritative layout,
packet format and install steps; the wire contract lives in
`driver/public/toustovac_virtual_mic_ioctl.h` and is shared by all three
components.

## Build

```powershell
# user-mode (works with the plain Windows SDK + MSVC as present here)
cmake -S native\virtual_mic -B build\virtual_mic
cmake --build build\virtual_mic --config Release

# kernel driver (requires the WDK include\km headers, §12)
msbuild /m native\virtual_mic\driver\ToustovacVirtualMic.vcxproj `
  /p:Configuration=Release /p:Platform=x64 /p:RunCodeAnalysis=true
infverif /w native\virtual_mic\package\ToustovacVirtualMic.inf
inf2cat /verbose /catalog:native\virtual_mic\package\ToustovacVirtualMic.cat `
  /usepnp:root /sign:200
```

## Install / uninstall

```powershell
ToustovacAudioInstall.exe -Install
ToustovacAudioInstall.exe -Uninstall
```

Idempotent, reverse-order rollback on failure, and the only package removal
covered is the one owned by this project.
