/*--
    minwavertstream.h - WaveRT stream for the single 48 kHz capture pin.

    Follows the modern sysvad pattern: the stream implements
    IMiniportWaveRTStreamNotification (the DPC-notification variant). It
    holds the DMA range, the notification event and the frame position; it
    pulls cleaned PCM from the CPCMRing shared with the control device.
--*/

#pragma once

#include <portcls.h>
#include <ks.h>
#include "pcm_ring.h"

class CMiniportWaveRT;
typedef CMiniportWaveRT *PCMiniportWaveRT;

class CMiniportWaveRTStream : public IMiniportWaveRTStreamNotification
{
public:
    static NTSTATUS Create(
        PCMiniportWaveRT                 WaveRt,
        CPCMRing*                        pRing,
        _In_   PPORTWAVERTSTREAM         PortStream,
        _Outptr_ CMiniportWaveRTStream** ppStream);

    // IUnknown
    virtual NTSTATUS QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object);
    virtual ULONG AddRef();
    virtual ULONG Release();

    // IMiniport
    virtual NTSTATUS Init(_In_ PUNKNOWN UnknownAdapter,
                          _In_ PRESOURCELIST ResourceList,
                          _In_ PPORT Port);

    // IMiniportWaveRTStream
    virtual NTSTATUS SetFormat(_In_ PKSDATAFORMAT DataFormat);
    virtual NTSTATUS SetState(_In_ KSSTATE State);
    virtual NTSTATUS GetPosition(_Out_ PKSAUDIO_POSITION Position);
    virtual NTSTATUS AllocateAudioBuffer(
        _In_  ULONG RequestedSize,
        _Out_ PMDL *AudioBufferMdl,
        _Out_ ULONG *ActualSize,
        _Out_ ULONG *OffsetFromFirstPage,
        _Out_ MEMORY_CACHING_TYPE *CacheType);
    virtual VOID FreeAudioBuffer(_In_opt_ PMDL AudioBufferMdl,
                                 _In_ ULONG BufferSize);
    virtual VOID GetHWLatency(_Out_ KSRTAUDIO_HWLATENCY *hwLatency);
    virtual NTSTATUS GetPositionRegister(_Out_ KSRTAUDIO_HWREGISTER *Register);
    virtual NTSTATUS GetClockRegister(_Out_ KSRTAUDIO_HWREGISTER *Register);

    // IMiniportWaveRTStreamNotification
    virtual NTSTATUS AllocateBufferWithNotification(
        _In_  ULONG NotificationCount,
        _In_  ULONG RequestedSize,
        _Out_ PMDL *AudioBufferMdl,
        _Out_ ULONG *ActualSize,
        _Out_ ULONG *OffsetFromFirstPage,
        _Out_ MEMORY_CACHING_TYPE *CacheType);
    virtual VOID FreeBufferWithNotification(_In_ PMDL AudioBufferMdl,
                                            _In_ ULONG BufferSize);
    virtual NTSTATUS RegisterNotificationEvent(_In_ PKEVENT NotificationEvent);
    virtual NTSTATUS UnregisterNotificationEvent(_In_ PKEVENT NotificationEvent);

    VOID Process(_In_ BOOLEAN fFlushPending);
    LONG GetProcessed() const { return m_nProcessed; }

protected:
    CMiniportWaveRTStream() :
        m_RefCount(1),
        m_nStreamID(0),
        m_pPhysicalDevice(NULL),
        m_pPortStream(NULL),
        m_pBufferMdl(NULL),
        m_nBufferSize(0),
        m_nSamplesPerFrame(480),
        m_nChannels(1),
        m_nMaxNumberOfFrames(0),
        m_pRing(NULL),
        m_pNotification(NULL),
        m_FirstProcessing(TRUE),
        m_nNextBufferTime(0),
        m_nProcessed(0)
        {}

    ~CMiniportWaveRTStream();

    LONG                            m_RefCount;
    ULONG                           m_nStreamID;
    PVOID                           m_pPhysicalDevice;
    PPORTWAVERTSTREAM               m_pPortStream;
    PMDL                            m_pBufferMdl;
    ULONG                           m_nBufferSize;          // bytes
    WORD                            m_nSamplesPerFrame;
    WORD                            m_nChannels;
    ULONG                           m_nMaxNumberOfFrames;
    CPCMRing*                       m_pRing;
    PKEVENT                         m_pNotification;
    BOOLEAN                         m_FirstProcessing;
    LONGLONG                        m_nNextBufferTime;      // 100 ns units
    volatile LONG                   m_nProcessed;           // frames delivered
    INT16                           m_Samples[480*2];       // DPC staging
};

typedef CMiniportWaveRTStream *PCMiniportWaveRTStream;
