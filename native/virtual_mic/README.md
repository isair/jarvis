# Toustovač audio stack — native/virtual_mic

Minimal Windows audio topology: one microphone capture endpoint plus its
service-side broker. Files map 1:1 to the plan; sources follow Microsoft
SysVAD (see `NOTICE` for the pinned upstream commit).

## Tree

```
native\virtual_mic\
├─ CMakeLists.txt        broker + installer (user-mode)              (see §12)
├─ NOTICE                SysVAD attribution + pinned upstream commit
├─ README.md             this file
├─ driver\               WDK driver package: capture-only endpoint
│  ├─ ToustovacVirtualMic.vcxproj    (Release|x64, codegen + inf2cat + /INTEGRITYCHECK)
│  ├─ ToustovacVirtualMic.inx        (hardware id ROOT\VIVERRA\TOUSTOVAC_CLEAN_MIC,
│  │                                  interface {A3F5D6B1-...-4417})
│  ├─ adapter.{h,cpp}                IAdapterCommon: topology + WaveRT ports
│  ├─ minip.{h,cpp}                  CPortInfo + CMiniportTopology
│  ├─ minwavert.{h,cpp}              CMiniportWaveRT (one stream)
│  ├─ minwavertstream.{h,cpp}        capture stream: DPC, QPC, ring reads
│  ├─ topology.{h,cpp}               node/connection/pin tables
│  ├─ formats.h                      48 kHz / mono / 16-bit PCM16 table
│  ├─ pcm_ring.{h,cpp}               50-frame (500 ms) SPSC ring + counters
│  ├─ control_device.{h,cpp}         SDDL-restricted control interface
│  ├─ public\toustovac_virtual_mic_ioctl.h   shared v1 wire format + GUID
│  └─ README.md
├─ broker\               ToustovacAudioBroker.exe (user-mode service)
│  ├─ service_main.cpp, broker.{h}, broker_service.cpp  SCM + state machine
│  ├─ driver_client.{h,cpp}          sole control-device handle
│  ├─ daemon_pipe.{h,cpp}            \\.\pipe\ToustovacCleanMic.v1 (message mode)
│  ├─ health.{h,cpp}                 1 s windows, heartbeat, silence fallback
│  └─ eventlog.{h,cpp}               System Event Log records
├─ installer\
│  └─ ToustovacAudioInstall.cpp      -Install/-Uninstall (no DevCon dependency)
└─ package\              staged artifacts (ToustovacAudio*.exe,
   ToustovacVirtualMic.{sys,inf,cat})
```

## Fixed identifiers

| Item | Value |
|---|---|
| Root devnode | `ROOT\VIVERRA\TOUSTOVAC_CLEAN_MIC` |
| Control interface (broker) | `{A3F5D6B1-2C8E-4A7F-9B1D-6E2F0C8A4417}` |
| Mic endpoint (IMMDevice) | `{B72E94C4-1D3F-5A86-BC0A-9D7421E3F2B0}` |
| Named pipe | `\\.\pipe\ToustovacCleanMic.v1` |
| Packet version | `TvmicPacketV1`, `struct_size = 52`, protocol `v1` |
| Ring capacity | 50 frames (500 ms) newest-real-time |

## Wire format (one packet)

`[u32 length][TvmicPacketV1 header (52 B)][PCM16 480×2 B]` — little-endian,
`qpc_100ns` in QueryPerformanceCounter ticks of the producer, `payload_crc32c`
=CRC32C of the PCM16 payload (0 for JSON control payloads: `{"protocol_version":
1,"frame_size":480,"sample_rate":48000,"channels":1}).` Sample layout is the
canonical 48 kHz / mono / 16-bit PCM only (no 24-bit / interleaved extras).

## Build (§12)

```powershell
cmake -S native\virtual_mic -B build\virtual_mic    # broker + installer
cmake --build build\virtual_mic --config Release
msbuild /m native\virtual_mic\driver\ToustovacVirtualMic.vcxproj `
  /p:Configuration=Release /p:Platform=x64 /p:RunCodeAnalysis=true
infverif /w native\virtual_mic\package\ToustovacVirtualMic.inf
inf2cat /verbose /catalog:.\<pkg>\ToustovacVirtualMic.cat `/usepnp:root` `/sign:200`
```

If the WDK `Include\km` set is absent, the driver project is delivered as
source; broker, installer and the `package\ToustovacVirtualMic.inf` stage
still complete (see the run output). DriverEntry is defined in
`adapter.cpp`.

## Runtime state machine (broker service, exactly per spec)

`starting -> driver_ready -> producer_connected -> streaming`; with each packet
that passes the header/CRC/sequence/size validation the ring is written and the
last-identity counters advance; on any loss or missed window the generation is
ended, the ring flushed and digital-zero silence is emitted
(`TVMIC_FLAG_SILENCE | TVMIC_FLAG_MUTED` on mute, `...|0` silent otherwise).

## Install/uninstall sequence

`ToustovacAudioInstall.exe -Install` verifies package files, stops a stale
broker, applies `DiInstallDriver` (idempotently: existing objects verified),
locates or creates `ROOT\VIVERRA\TOUSTOVAC_CLEAN_MIC`, creates + starts the
`ToustovacAudioBroker` service and confirms the endpoint via the daemon's
publisher status. `ToustovacAudioInstall.exe -Uninstall` = stop → delete
service → remove devnode → remove the owned package only. First-failure
rollback runs in reverse creation order; the newest signed package overwrites
the older one only once its healthy endpoint is confirmed.
