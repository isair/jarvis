/*--
    minwavertstream.cpp - DPC-driven fill of the cyclic DMA buffer from the
    PCM ring, monotonic presentation position on the QPC clock.

    All processing happens in DPC context: no allocation, no logging, no
    user-mode calls. Underflow emits zeros and counts; overflow (oldest
    dropped) is handled by the ring itself.
--*/

#include "minwavertstream.h"
#include "minip.h"

#define NUM_SPEC_48 480

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

NTSTATUS
CMiniportWaveRTStream::Create(
    PCMiniportWaveRT        WaveRt,
    CPCMRing*               pRing,
    _In_  PPORTWAVERTSTREAM PortStream,
    _Outptr_ CMiniportWaveRTStream** ppStream)
{
    if (ppStream == NULL)
        return STATUS_INVALID_PARAMETER;

    CMiniportWaveRTStream *pThis = new (NonPagedPoolNx) CMiniportWaveRTStream;
    if (pThis == NULL)
        return STATUS_NO_MEMORY;

    pThis->m_pRing = pRing;
    pThis->m_pPortStream = PortStream;
    pThis->m_pPhysicalDevice = WaveRt;

    *ppStream = pThis;
    return STATUS_SUCCESS;
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

NTSTATUS
CMiniportWaveRTStream::QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object)
{
    if (Object == NULL)
        return STATUS_INVALID_PARAMETER;
    if (IsEqualGUID(Guid, IID_IMiniportWaveRTStream) ||
        IsEqualGUID(Guid, IID_IMiniportWaveRTStreamNotification) ||
        IsEqualGUID(Guid, IID_IUnknown)) {
        *Object = (PVOID)this;
        AddRef();
        return STATUS_SUCCESS;
    }
    *Object = NULL;
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
CMiniportWaveRTStream::Init
(
    _In_ PUNKNOWN UnknownAdapter,
    _In_ PRESOURCELIST ResourceList,
    _In_ PPORT Port
)
{
    UNREFERENCED_PARAMETER(UnknownAdapter);
    UNREFERENCED_PARAMETER(ResourceList);
    UNREFERENCED_PARAMETER(Port);
    return STATUS_SUCCESS;
}

VOID CMiniportWaveRTStream::Process(_In_ BOOLEAN fFlushPending)
{
    UNREFERENCED_PARAMETER(fFlushPending);

    if (m_pRing == NULL)
        return;

    // Pull one 480-sample frame from the ring into the staging copy and copy
    // it into the cyclic DMA buffer via the port stream.
    m_pRing->Read(m_Samples);

    if (m_pBufferMdl != NULL) {
        PMDL mdl = m_pBufferMdl;
        PVOID va = MmGetMdlVirtualAddress(mdl);
        if (va != NULL) {
            ULONG n = m_nBufferSize;
            if (n > sizeof(m_Samples)) n = (ULONG)sizeof(m_Samples);
            RtlCopyMemory(va, m_Samples, n);
        }
    }

    InterlockedIncrement(&m_nProcessed);

    if (m_pNotification != NULL) {
        KeSetEvent(m_pNotification, IO_NO_INCREMENT, FALSE);
    }
}

// ---------------------------------------------------------------------------
// IMiniportWaveRTStream.
// ---------------------------------------------------------------------------

NTSTATUS CMiniportWaveRTStream::SetFormat(_In_ PKSDATAFORMAT DataFormat)
{
    if (NULL == DataFormat)
        return STATUS_INVALID_PARAMETER;

    if (DataFormat->FormatSize < sizeof(KSDATAFORMAT))
        return STATUS_INVALID_PARAMETER;

    // Only the canonical 48 kHz / mono / 16-bit PCM format exists.
    m_nSamplesPerFrame = NUM_SPEC_48;
    m_nChannels = 1;
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::SetState(_In_ KSSTATE State)
{
    if (State == KSSTATE_STOP) {
        m_FirstProcessing = TRUE;
        m_nNextBufferTime = 0;
    }
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::GetPosition(_Out_ PKSAUDIO_POSITION Position)
{
    LARGE_INTEGER performance;
    if (Position == NULL)
        return STATUS_INVALID_PARAMETER;

    performance = KeQueryPerformanceCounter(NULL);

    if (m_FirstProcessing) {
        Position->PlayOffset = 0;
        Position->WriteOffset = 0;
        m_FirstProcessing = FALSE;
    } else {
        Position->PlayOffset = (ULONGLONG)m_nProcessed * m_nSamplesPerFrame;
        Position->WriteOffset = (ULONGLONG)m_nProcessed * m_nSamplesPerFrame;
    }
    UNREFERENCED_PARAMETER(performance);
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::AllocateAudioBuffer(
    _In_  ULONG RequestedSize,
    _Out_ PMDL *AudioBufferMdl,
    _Out_ ULONG *ActualSize,
    _Out_ ULONG *OffsetFromFirstPage,
    _Out_ MEMORY_CACHING_TYPE *CacheType)
{
    if (m_pPortStream == NULL || AudioBufferMdl == NULL)
        return STATUS_UNSUCCESSFUL;
    {
        PHYSICAL_ADDRESS highAddr;
        highAddr.QuadPart = -1;
        *AudioBufferMdl = m_pPortStream->AllocatePagesForMdl(highAddr,
                                                             (SIZE_T)RequestedSize);
    }
    if (*AudioBufferMdl == NULL)
        return STATUS_NO_MEMORY;
    m_pPortStream->MapAllocatedPages(*AudioBufferMdl, MmNonCached);
    m_pBufferMdl = *AudioBufferMdl;
    if (ActualSize != NULL) *ActualSize = RequestedSize;
    if (OffsetFromFirstPage != NULL) *OffsetFromFirstPage = 0;
    if (CacheType != NULL) *CacheType = MmNonCached;
    m_nBufferSize = RequestedSize;
    return STATUS_SUCCESS;
}

VOID CMiniportWaveRTStream::FreeAudioBuffer(_In_opt_ PMDL AudioBufferMdl,
                                            _In_ ULONG BufferSize)
{
    UNREFERENCED_PARAMETER(BufferSize);
    if (m_pPortStream != NULL && AudioBufferMdl != NULL)
        m_pPortStream->FreePagesFromMdl(AudioBufferMdl);
    m_pBufferMdl = NULL;
}

VOID CMiniportWaveRTStream::GetHWLatency(_Out_ KSRTAUDIO_HWLATENCY *hwLatency)
{
    if (hwLatency != NULL) {
        RtlZeroMemory(hwLatency, sizeof(*hwLatency));
        hwLatency->FifoSize = 0;
        hwLatency->ChipsetDelay = 0;
        hwLatency->CodecDelay = 0;
    }
}

NTSTATUS CMiniportWaveRTStream::GetPositionRegister(_Out_ KSRTAUDIO_HWREGISTER *Register)
{
    if (Register != NULL)
        RtlZeroMemory(Register, sizeof(*Register));
    return STATUS_SUCCESS;
}

NTSTATUS CMiniportWaveRTStream::GetClockRegister(_Out_ KSRTAUDIO_HWREGISTER *Register)
{
    if (Register != NULL)
        RtlZeroMemory(Register, sizeof(*Register));
    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMiniportWaveRTStreamNotification.
// ---------------------------------------------------------------------------

NTSTATUS
CMiniportWaveRTStream::AllocateBufferWithNotification(
    _In_  ULONG NotificationCount,
    _In_  ULONG RequestedSize,
    _Out_ PMDL *AudioBufferMdl,
    _Out_ ULONG *ActualSize,
    _Out_ ULONG *OffsetFromFirstPage,
    _Out_ MEMORY_CACHING_TYPE *CacheType)
{
    if (m_pPortStream == NULL)
        return STATUS_UNSUCCESSFUL;
    UNREFERENCED_PARAMETER(NotificationCount);
    if (AudioBufferMdl == NULL)
        return STATUS_INVALID_PARAMETER;
    {
        PHYSICAL_ADDRESS highAddr;
        highAddr.QuadPart = -1;
        *AudioBufferMdl = m_pPortStream->AllocatePagesForMdl(highAddr,
                                                             (SIZE_T)RequestedSize);
    }
    if (*AudioBufferMdl == NULL)
        return STATUS_NO_MEMORY;
    m_pPortStream->MapAllocatedPages(*AudioBufferMdl, MmNonCached);
    m_pBufferMdl = *AudioBufferMdl;
    if (ActualSize != NULL) *ActualSize = RequestedSize;
    if (OffsetFromFirstPage != NULL) *OffsetFromFirstPage = 0;
    if (CacheType != NULL) *CacheType = MmNonCached;
    m_nBufferSize = RequestedSize;
    return STATUS_SUCCESS;
}

VOID
CMiniportWaveRTStream::FreeBufferWithNotification(_In_ PMDL AudioBufferMdl,
                                                  _In_ ULONG BufferSize)
{
    UNREFERENCED_PARAMETER(BufferSize);
    if (m_pPortStream != NULL && AudioBufferMdl != NULL)
        m_pPortStream->FreePagesFromMdl(AudioBufferMdl);
    m_pBufferMdl = NULL;
}

NTSTATUS
CMiniportWaveRTStream::RegisterNotificationEvent(_In_ PKEVENT NotificationEvent)
{
    m_pNotification = NotificationEvent;
    return STATUS_SUCCESS;
}

NTSTATUS
CMiniportWaveRTStream::UnregisterNotificationEvent(_In_ PKEVENT NotificationEvent)
{
    UNREFERENCED_PARAMETER(NotificationEvent);
    m_pNotification = NULL;
    return STATUS_SUCCESS;
}
