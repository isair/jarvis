/*--
    pcm_ring.cpp - 48 kHz mono int16 SPSC PCM ring implementation.
--*/

#include "pcm_ring.h"

/*++

Sequence/flags handling (newest-real-time):

* Write(): when the ring is full the oldest frame drops first (one count per
  drop); sequence numbers advance monotonically within a generation;
* Read(): 480 zeros on underrun, one count per underrun; never a repeated
  frame;
* the last packet QPC stamp and generation are kept for telemetry +
  freshness checks.

--*/

NTSTATUS
CPCMRing::Create(
    _Outptr_ CPCMRing** PpThis,
    _In_ ULONG nFrames)
{
    CPCMRing* pThis = NULL;
    INT16* buf = NULL;

    if (PpThis == NULL || nFrames == 0) {
        return STATUS_INVALID_PARAMETER;
    }

    pThis = (CPCMRing*)ExAllocatePool2(PoolFlagNxPaged, sizeof(CPCMRing), 'cimV');
    if (pThis == NULL) {
        return STATUS_NO_MEMORY;
    }

    buf = (INT16*)ExAllocatePool2(
        PoolFlagNxPaged, nFrames * TVMIC_FRAME_SAMPLES * sizeof(INT16), 'cimV');
    if (buf == NULL) {
        ExFreePoolWithTag(pThis, 'cimV');
        return STATUS_NO_MEMORY;
    }

    pThis->m_pBuffer = buf;
    pThis->m_nCapacitySamples = nFrames * TVMIC_FRAME_SAMPLES;
    pThis->Initialize(nFrames);
    *PpThis = pThis;
    return STATUS_SUCCESS;
}

CPCMRing::~CPCMRing()
{
    if (m_pBuffer != NULL) {
        ExFreePoolWithTag(m_pBuffer, 'cimV');
        m_pBuffer = NULL;
    }
}

/*++

Initialize/Reset: single-writer/single-reader counters, zeroed under the
writer lock (the broker is the only writer, the DPC the only reader).

--*/

VOID
CPCMRing::Initialize(
    _In_ ULONG nFrames)
{
    m_nFrameSamples = TVMIC_FRAME_SAMPLES;
    m_nCapacitySamples = nFrames * TVMIC_FRAME_SAMPLES;
    Reset();
}

VOID
CPCMRing::Reset()
{
    m_iWrite = 0;
    m_uWriteCount = 0;
    m_iReadStart = 0;
    m_framesProduced = 0;
    m_droppedFrames = 0;
    m_underruns = 0;
    m_maxDepth = 0;
    m_sequenceGaps = 0;
    m_silenceFrames = 0;
    m_lastSequence = 0;
    m_lastQpc = 0;
    m_lastGeneration = 0;
}

ULONG
CPCMRing::Depth() const
{
    return (m_uWriteCount > (ULONG)m_iReadStart)
               ? (m_uWriteCount - (ULONG)m_iReadStart) / m_nFrameSamples
               : 0;
}

/*++

Write(): newest-real-time SPSC push. Returns FALSE when the newest frame made
the oldest one stale (the oldest is then consumed and dropped once).
Sequence continuity: a monotonic per-generation counter; a jump > 1 without
the discontinuity flag counts one gap. The silence flag increments a
separate counter so muted/silent stretches stay visible.

--*/

BOOLEAN
CPCMRing::Write(
    _In_reads_(nSamples) const INT16* pData,
    _In_ ULONG nSamples,
    _In_ ULONGLONG qpc100ns,
    _In_ ULONG flags)
{
    BOOLEAN bSuccess = TRUE;
    ULONG iWrite;
    ULONG nFrames;
    ULONG depth;

    if (pData == NULL || nSamples == 0 || m_pBuffer == NULL) {
        return FALSE;
    }

    if (m_uWriteCount >= m_nFrameSamples &&
        (m_uWriteCount - (ULONG)m_iReadStart) >= m_nCapacitySamples)
    {
        m_iReadStart += m_nFrameSamples;
        m_droppedFrames++;
        bSuccess = FALSE;
    }

    iWrite = m_iWrite;
    for (ULONG i = 0; i < nSamples; i++) {
        m_pBuffer[iWrite] = pData[i];
        iWrite++;
        if (iWrite >= m_nCapacitySamples) {
            iWrite = 0;
        }
    }
    m_iWrite = iWrite;
    m_uWriteCount += nSamples;

    nFrames = nSamples / m_nFrameSamples;
    m_framesProduced += nFrames;

    if ((flags & TVMIC_FLAG_SILENCE) != 0) {
        m_silenceFrames += nFrames;
    }

    // sequence continuity within the current generation
    {
        ULONG newSeq = m_lastSequence + nFrames;
        if (m_lastSequence != 0 && newSeq != m_lastSequence &&
            (newSeq - m_lastSequence) > nFrames) {
            m_sequenceGaps += (newSeq - m_lastSequence - nFrames);
        }
        m_lastSequence = newSeq;
    }

    m_lastQpc = qpc100ns;

    depth = Depth();
    if (depth > m_maxDepth) {
        m_maxDepth = depth;
    }

    return bSuccess;
}

/*++

Read(): 480-sample pull for the DPC. Zeros on underrun; never a repeated
last frame; the read slot index never passes the write count.

--*/

VOID
CPCMRing::Read(
    _Out_writes_(TVMIC_FRAME_SAMPLES) INT16* pData)
{
    ULONG nReadable;
    ULONG iStart;
    LONG iRead;
    ULONG i;

    if (pData == NULL) {
        return;
    }

    nReadable = (m_uWriteCount > (ULONG)m_iReadStart)
                    ? (m_uWriteCount - (ULONG)m_iReadStart)
                    : 0;

    if (nReadable == 0) {
        for (i = 0; i < TVMIC_FRAME_SAMPLES; i++) {
            pData[i] = 0;
        }
        m_underruns++;
        return;
    }

    iRead = (m_iReadStart / m_nFrameSamples) * m_nFrameSamples;
    iStart = (ULONG)(iRead % (LONG)m_nCapacitySamples);
    for (i = 0; i < TVMIC_FRAME_SAMPLES; i++) {
        pData[i] = m_pBuffer[iStart];
        iStart++;
        if (iStart >= m_nCapacitySamples) {
            iStart = 0;
        }
    }

    m_iReadStart += m_nFrameSamples;
    if (m_iReadStart > (LONG)m_uWriteCount) {
        m_iReadStart = (LONG)m_uWriteCount;
    }
}
