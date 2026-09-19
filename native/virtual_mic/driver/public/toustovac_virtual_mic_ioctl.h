/*--

Module Name:

    toustovac_virtual_mic_ioctl.h

Abstract:

    Shared wire format of the Toustovač Clean Microphone endpoint. One
    versioned packet header (little-endian, 52 bytes) is used on every
    layer: the in-process CleanAudioBus, the named-pipe broker messages and
    the kernel-side PCM ring. The 100 ns clock is the QueryPerformanceCounter
    unit of the packet producer (the kernel native QPC unit matches).

    Named pipe:  \\.\pipe\ToustovacCleanMic.v1
    Framing:     [u32 total_length][TvmicPacketV1 header][PCM16 payload]

    The header uses the Windows fixed-width names (UINT32/UINT64), so both
    the KM (ntdef) and UM (windows.h) environments agree.

    The header must be 52 bytes in both environments: 4+4+8+8+8+4+2+2+4+4+4
    = 52 with 1-byte alignment. #pragma pack(1) makes this deterministic.

--*/

#ifndef TOUSTOVAC_VIRTUAL_MIC_IOCTL_H
#define TOUSTOVAC_VIRTUAL_MIC_IOCTL_H

#ifdef __cplusplus
extern "C" {
#endif

#define TVMIC_PIPE_NAME L"\\\\.\\pipe\\ToustovacCleanMic.v1"
#define TVMIC_PROTOCOL_VERSION 1u
#define TVMIC_SAMPLE_RATE 48000u
#define TVMIC_CHANNELS 1u
#define TVMIC_BITS_PER_SAMPLE 16u
#define TVMIC_FRAME_SAMPLES 480u

/* Wire sizes: 52-byte header + optional 2-byte-per-sample PCM16 payload. */
#define TVMIC_PACKET_HEADER_SIZE 52u
#define PACKET_HEADER_SIZE TVMIC_PACKET_HEADER_SIZE
#define TVMIC_PCM16_SAMPLES_PER_PACKET TVMIC_FRAME_SAMPLES
#define TVMIC_MAX_PACKET_BYTES (TVMIC_PACKET_HEADER_SIZE + (TVMIC_FRAME_SAMPLES * 2u))

/* Packet flags: fixed bit positions shared by driver, broker and daemon. */
#define TVMIC_FLAG_LOCAL 0x00000001u         /* source kind: local USB lane      */
#define TVMIC_FLAG_SILENCE 0x00000002u       /* payload is explicit digital zero */
#define TVMIC_PACKET_FLAG_DISCONTINUITY 0x00000004u /* new gen or gap            */
#define TVMIC_FLAG_MUTED 0x00000008u         /* muted: payload is silence        */
#define TVMIC_FLAG_FIRST_OF_GENERATION 0x00000010u

/* One canonical packet. Field order defines the wire layout (little-endian,
 * 52 bytes). struct_size must equal 52. QPC is in 100 ns units.
 * payload_crc32c is 0 for JSON control payloads. */
#pragma pack(push, 1)
typedef struct _TvmicPacketV1 {
    UINT32 struct_size;
    UINT32 protocol_version;
    UINT64 producer_generation;
    UINT64 sequence;
    UINT64 qpc_100ns;
    UINT32 sample_rate;       /* exactly 48000 */
    UINT16 channels;          /* exactly 1     */
    UINT16 bits_per_sample;   /* exactly 16    */
    UINT32 sample_count;      /* int16 samples (typically 480); 0 for
                               * JSON control payloads */
    UINT32 flags;             /* TVMIC_FLAG_* / TVMIC_PACKET_FLAG_* ORed    */
    UINT32 payload_crc32c;    /* CRC32C (Castagnoli) of the PCM16 payload   */
} TvmicPacketV1, *PTvmicPacketV1;
#pragma pack(pop)

/* ---------------------------------------------------------------------------
 * Private device interface (the control device object the broker opens).
 * ------------------------------------------------------------------------ */

#define GUID_DEVINTERFACE_TOUSTOVAC_VIRTUAL_MIC_CONTROL \
    {0xA3F5D6B1, 0x2C8E, 0x4A7F, {0x9B, 0x1D, 0x6E, 0x2F, 0x0C, 0x8A, 0x44, 0x17}}

#define IOCTL_TVMIC_NEGOTIATE \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800u, METHOD_BUFFERED, FILE_READ_ACCESS | FILE_WRITE_ACCESS)
#define IOCTL_TVMIC_SET_MUTE \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801u, METHOD_BUFFERED, FILE_READ_ACCESS | FILE_WRITE_ACCESS)
#define IOCTL_TVMIC_BEGIN_GENERATION \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x802u, METHOD_BUFFERED, FILE_READ_ACCESS | FILE_WRITE_ACCESS)
#define IOCTL_TVMIC_END_GENERATION \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x803u, METHOD_BUFFERED, FILE_READ_ACCESS | FILE_WRITE_ACCESS)
#define IOCTL_TVMIC_QUERY_STATUS \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x804u, METHOD_BUFFERED, FILE_READ_ACCESS | FILE_WRITE_ACCESS)
#define IOCTL_TVMIC_WRITE_FRAMES \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x805u, METHOD_BUFFERED, FILE_READ_ACCESS | FILE_WRITE_ACCESS)

/* Negotiation payload: producer -> driver, canonical 48 kHz / mono / 16-bit. */
typedef struct _TvmicInit {
    UINT32 struct_size;       /* 28 */
    UINT32 protocol_version;  /* TVMIC_PROTOCOL_VERSION */
    UINT32 sample_rate;       /* 48000 */
    UINT32 channels;          /* 1 */
    UINT32 bits_per_sample;   /* 16 */
    UINT32 frame_samples;     /* 480 */
    UINT64 producer_generation;
} TvmicInit, *PTvmicInit;

/* Query status: driver -> producer counters (64 bytes, packed layout). */
#pragma pack(push, 1)
typedef struct _TvmicCtlStatus {
    UINT32 struct_size;                /* 64 */
    UINT32 protocol;                   /* 1  */
    UINT32 sample_rate;
    UINT32 channels;
    UINT32 bits_per_sample;
    UINT32 active_capture_clients;     /* IMMDEVICE count seen by the engine */
    UINT32 frame_samples;
    UINT32 ring_capacity_frames;       /* 50 == 500 ms */
    UINT32 ring_depth_frames;
    UINT64 last_sequence;
    UINT64 last_timestamp_100ns;
    UINT64 frames_produced;
    UINT64 silence_frames;
    UINT64 driver_underflows;
    UINT64 driver_overflows;
    UINT64 rejected_packets;
    UINT64 stale_packets;
    UINT64 sequence_gaps;
    UINT64 max_ring_depth_frames;
    UINT64 producer_generation;
} TvmicCtlStatus, *PTvmicCtlStatus;
#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* TOUSTOVAC_VIRTUAL_MIC_IOCTL_H */
