/*--
    pcm_ring.h - tiny SPSC PCM ring shared by the WaveRT DPC and the named-pipe
    broker that feeds it. Int16 samples, 48 kHz / mono, 50-frame (500 ms)
    capacity. The ring also tracks the identity of the last packet
    (generation/sequence/QPC/flags) for continuity checks and telemetry
    counters for producer / driver / stale / gap accounting.
--*/

#pragma once

#include <ntddk.h>
#include "public/toustovac_virtual_mic_ioctl.h"

class CPCMRing
{
public:
    static NTSTATUS Create(_Outptr_ CPCMRing** PpThis, _In_ ULONG nFrames);
    ~CPCMRing();

    // Writer side (control device dispatch). Returns FALSE when the newest
    // frame makes an older one stale: the oldest is then dropped and the
    // overflow counter increments once.
    BOOLEAN Write(_In_reads_(nSamples) const INT16* pData, _In_ ULONG nSamples,
                  _In_ ULONGLONG qpc100ns, _In_ ULONG flags);
    // Reader side (DPC Process). Fills 480 samples with zeros when empty.
    VOID Read(_Out_writes_(TVMIC_FRAME_SAMPLES) INT16* pData);

    ULONG CapacityFrames() const { return m_nCapacitySamples / m_nFrameSamples; }
    ULONG Depth() const;
    ULONG FramesProduced() const { return m_framesProduced; }
    ULONG SilenceFrames() const { return m_silenceFrames; }
    ULONG FramesDropped() const { return m_droppedFrames; }       // real overruns
    ULONG Underruns() const { return m_underruns; }               // zeros emitted
    UINT64 StalePackets() const { return (UINT64)m_droppedFrames; }
    UINT64 SequenceGaps() const { return m_sequenceGaps; }
    ULONG LastSequence() const { return m_lastSequence; }
    ULONGLONG LastQpc() const { return m_lastQpc; }
    UINT64 Generation() const { return (UINT64)m_lastGeneration; }
    UINT64 MaxDepth() const { return m_maxDepth; }

    VOID Initialize(_In_ ULONG nFrames);
    VOID Reset();
    VOID SetGeneration(_In_ UINT64 generation) { m_lastGeneration = generation; }

private:
    CPCMRing() :
        m_pBuffer(NULL), m_nCapacitySamples(0), m_nFrameSamples(TVMIC_FRAME_SAMPLES),
        m_iWrite(0), m_uWriteCount(0), m_iReadStart(0),
        m_framesProduced(0), m_droppedFrames(0), m_underruns(0),
        m_maxDepth(0), m_sequenceGaps(0), m_silenceFrames(0),
        m_lastSequence(0), m_lastQpc(0), m_lastGeneration(0) {}

    INT16*             m_pBuffer;
    ULONG              m_nCapacitySamples;   // samples total (capacity * 480)
    ULONG              m_nFrameSamples;      // per-frame count (480)
    volatile ULONG     m_iWrite;             // next sample index to write
    volatile ULONG     m_uWriteCount;        // total samples written
    volatile LONG      m_iReadStart;         // oldest unread sample index
    volatile ULONG     m_framesProduced;     // frames written
    volatile ULONG     m_droppedFrames;      // real overruns (oldest dropped)
    volatile ULONG     m_underruns;          // real underruns (zeros emitted)
    volatile ULONG     m_maxDepth;           // deepest observed ring depth
    volatile ULONG     m_sequenceGaps;       // missing sequence numbers
    volatile ULONG     m_silenceFrames;      // TVMIC_FLAG_SILENCE frames
    volatile ULONG     m_lastSequence;       // sequence of the newest packet
    volatile ULONGLONG m_lastQpc;            // QPC stamp of the newest packet
    volatile ULONGLONG m_lastGeneration;     // generation of the newest packet
};
