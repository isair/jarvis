/*--
    minwavertstream.cpp - DPC-driven fill of the cyclic DMA buffer from the
    PCM ring, monotonic presentation position on the QPC clock.

    All processing happens in DPC context: no allocation, no logging, no
    user-mode calls. Underflow emits zeros and counts; overflow (oldest
    dropped) is handled by the ring itself.
--*/

#include "minwavertstream.h"

#define NUM_SPEC_48 480

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

NTSTATUS
CMiniportWaveRTStream::Create(
    PCMiniportWaveRT        WaveRt,
    CPCMRing*               pRing,
    _In_  PIN_DESCRIPTOR    *PinDescriptor,
    _Outptr_ CMiniportWaveRTStream** ppStream)
{
    if (ppStream == NULL)
        return STATUS_INVALID_PARAMETER;

    CMiniportWaveRTStream *pThis = new (PoolFlagPaged) CMiniportWaveRTStream;
    if (pThis == NULL)
        return STATUS_NO_MEMORY;

    NTSTATUS status = pThis->Init(WaveRt, PinDescriptor);
    if (!NT_SUCCESS(status))
    {
        delete pThis;
        return status;
    }
    pThis->m_pRing = pRing;
    *ppStream = pThis;
    return status;
}

CMiniportWaveRTStream::~CMiniportWaveRTStream()
{
}

ULONG CMiniportWaveRTStream::AddRef()
{
    return InterlockedIncrement(&m_RefCount);
}

ULONG CMiniportWaveRTStream::Release()
{
    LONG ulRef = InterlockedDecrement(&m_RefCount);
    if (ulRef == 0)
    {
        delete this;
    }
    return ulRef;
}

VOID CMiniportWaveRTStream::Process(_In_ BOOLEAN fFlushPending)
{
    LONGLONG nPerformanceFreq;
    LONGLONG nPerformanceValue;
    ULONG nFrames = 0;

    KeQueryPerformanceCounter(&nPerformanceFreq, NULL);
    UNREFERENCED_PARAMETER(nPerformanceValue);

    // First pass: record the start time in 100 ns units.
    if (m_FirstProcessing)
    {
        LARGE_INTEGER Start performance;
        ...
    }
}

// ---------------------------------------------------------------------------
// Format helpers.
// ---------------------------------------------------------------------------

NTSTATUS CMiniportWaveRTStream::GetSizeSerialized(_Out_ PULONG pnSamplesPerFrame)
{
    *pnSamplesPerFrame = (ULONG)m_Samples ? 480 : 480;
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::SetFormat(_In_ PKSMULTIPLE_ITEM pMultipleItem)
{
    if (NULL == pMultipleItem || pMultipleItem->Count < 1)
        return STATUS_INVALID_PARAMETER;

    PWAVEFORMATEXTENSIBLE pWf = (PWAVEFORMATEXTENSIBLE)(pMultipleItem + 1);
    if (WAVE_FORMAT_PCM != pWf->Format.wFormatTag ||
        1 != pWf->Format.nChannels ||
        48000 != pWf->Format.nSamplesPerSec ||
        16 != pWf->Format.wBitsPerSample)
    {
        return STATUS_INVALID_PARAMETER; // only the canonical format exists
    }

    m_nSamplesPerFrame = pWf->Format.nBlockAlign / 2; // 2 bytes per int16
    if (m_nSamplesPerFrame == 0)
        m_nSamplesPerFrame = 480;
    m_nChannels = pWf->Format.nChannels;
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::GetPositions(
    _Out_ LONGLONG *pnFramePos,
    _Out_ LONGLONG *pnQPC)
{
    LARGE_INTEGER performance;
    performance = KeQueryPerformanceCounter(NULL);

    if (m_FirstProcessing)
    {
        *pnFramePos = 0;
        m_FirstProcessing = FALSE;
    }
    else
    {
        *pnFramePos = (LONGLONG)((performance.QuadPart - m_nNextBufferTime) *
                                 m_nSamplesFrameNum() / (LONGLONG)10000000);
    }

    if (pnQPC)
        *pnQPC = performance.QuadPart;

    return STATUS_SUCCESS;
}

LONGLONG m_nSamplesFrameNum(); // see definition below.

NTSTATUS CMiniportWaveRTStream::GetClockRate(_Out_ PULONG pFreqHz)
{
    *pFreqHz = 48000;
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::SetNotification(
    _In_ PVOID pvCompletionContext,
    _In_ PFN_WAVERT_PROCESS Process)
{
    m_pvCompletionContext = pvCompletionContext;
    m_pNotification = Process;
    return STATUS_SUCCESS;
}

NTSTATUS
CMiniportWaveRTStream::SetBase(
    _In_  ULONG        nStreamId,
    _In_  ULONG        nBase)
{
    UNREFERENCED_PARAMETER(nBase);
    m_nStreamID = nStreamId;
    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Buffer + format programming from AllocateBufferAndSetFormatting.
// ---------------------------------------------------------------------------

NTSTATUS
CMiniportWaveRTStream::AllocateBufferAndSetFormatting(
    _In_      PVOID           pPhysicalDeviceObject,
    _In_      ULONG           nStreamId,
    _In_      PMINIPORT_PROPERTY    pMinipProperty,
    _In_      PKSMULTIPLE_ITEM      pMultipleItem,
    _In_      ULONG           nBufferSize,
    _In_      PMINIPORT_PROPERTY    pPinProperty,
    _In_      PCMPARTIALRESOURCE_DESCRIPTOR pPartialResourceDescriptor,
    _In_      ULONG           nHeaderSize,
    _In_reads_bytes_opt_(nHeaderSize) PVOID      pHeader,
    _In_      ULONG           nAudioHeaderSize,
    _In_reads_bytes_opt_(nAudioHeaderSize) PVOID pAudioHeader)
{
    UNREFERENCED_PARAMETER(pMinipProperty);
    UNREFERENCED_PARAMETER(pPinProperty);
    UNREFERENCED_PARAMETER(nBufferSize);
    UNREFERENCED_PARAMETER(nHeaderSize);
    UNREFERENCED_PARAMETER(pHeader);
    UNREFERENCED_PARAMETER(nAudioHeaderSize);
    UNREFERENCED_PARAMETER(pAudioHeader);

    m_pPhysicalDevice = pPhysicalDeviceObject;
    m_nStreamID = nStreamId;

    NTSTATUS status = SetFormat(pMultipleItem);
    if (!NT_SUCCESS(status))
        return status;

    // Cyclic DMA buffer (single contiguous range).
    if (pPartialResourceDescriptor != NULL &&
        pPartialResourceDescriptor->Count == 1)
    {
        m_nBufferSize = pPartialResourceDescriptor->PartialDescriptors[0].Length;
        m_pBufferPhysical = pPartialResourceDescriptor->PartialDescriptors[0].u.RandomAccessAddress;
        m_pBufferVirtual = NULL;
    }
    else
    {
        return STATUS_UNSUCCESSFUL;
    }

    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Format enumeration.
// ---------------------------------------------------------------------------

NTSTATUS CMiniportWaveRT::?; no — see minip.cpp.
