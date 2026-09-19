/*--
    minwavertstream.h - WaveRT stream for the single 48 kHz capture pin.

    Follows the sysvad pattern: the stream object holds
    the DMA range, the DPC notification bookkeeping and the frame position;
    it pulls cleaned PCM from the CPCMRing shared with the control device.
--*/

#pragma once

#include <portcls.h>
#include <ks.h>
#include "pcm_ring.h"

typedef struct _DMABuffer
{
    PVOID               VirtualAddress;
    ULONG               PhysicalAddressLower32Bit;
    ULONGLONG           PhysicalAddress;
    ULONG               nContiguousRange;
    ULONG               nDescriptorRange;
    // The contiguous scatter/gather list for the cyclic DMA buffer.
    PHYSICAL_ADDRESS    PhysicalAddressList[1];
} DMABuffer;

class CMiniportWaveRTStream : public IMiniportWaveRTStream
{
public:
    static NTSTATUS Create(
        PCMiniportWaveRT                WaveRt,
        CPCMRing*                        pRing,
        _In_   PIN_DESCRIPTOR            *PinDescriptor,
        _Outptr_ CMiniportWaveRTStream** ppStream);

    // IUnknown
    virtual ULONG AddRef();
    virtual ULONG Release();

    // IMiniportWaveRTStream
    virtual NTSTATUS
    GetSizeSerialized (
        _Out_ PULONG          pnSamplesPerFrame
    );

    virtual NTSTATUS
    SetFormat (
        _In_  PKSMULTIPLE_ITEM    pMultipleItem
    );

    virtual NTSTATUS
    GetPositions (
        _Out_     LONGLONG *            pnFramePos,
        _Out_     LONGLONG *            pnQPC
    );

    virtual NTSTATUS
    GetClockRate (
        _Out_     PULONG                pFreqHz
    );

    virtual NTSTATUS
    SetNotification (
        _In_      PVOID                pvCompletionContext,
        _In_      PFN_WAVERT_PROCESS   Process
    );

    virtual NTSTATUS
    SetBase (
        _In_      ULONG                nStreamId,
        _In_      ULONG                nBase
    );

    virtual NTSTATUS
    GetFormat (
        _In_      ULONG                nFormatIndex,
        _Out_     PULONG               pnFormatSize,
        _Out_writes_bytes_(*pnFormatSize) PVOID pFormat
    );

    virtual NTSTATUS
    AllocateBufferAndSetFormatting (
        _In_      PVOID                pPhysicalDeviceObject,
        _In_      ULONG                nStreamId,
        _In_      PMINIPORT_PROPERTY   pMinipProperty,
        _In_      PKSMULTIPLE_ITEM     pMultipleItem,
        _In_      ULONG                nBufferSize,
        _In_      PMINIPORT_PROPERTY   pPinProperty,
        _In_      PCMPARTIALRESOURCE_DESCRIPTOR pPartialResourceDescriptor,
        _In_      ULONG                nHeaderSize,
        _In_reads_bytes_opt_(nHeaderSize) PVOID  pHeader,
        _In_      ULONG                nAudioHeaderSize,
        _In_reads_bytes_opt_(nAudioHeaderSize) PVOID pAudioHeader
    );

    NTSTATUS
    Init (
        _In_  PCMiniportWaveRT        WaveRt,
        _In_  PPIN_DESCRIPTOR         pPinDescriptor
    );

    VOID Process(_In_ BOOLEAN fFlushPending);

    LONG GetProcessed() const;

protected:
    CMiniportWaveRTStream() :
        m_RefCount(1),
        m_nStreamID(0),
        m_pPhysicalDevice(NULL),
        m_pPinDescriptor(NULL),
        m_nSamplesPerFrame(0),
        m_nChannels(1),
        m_nMaxNumberOfFrames(0),
        m_nBufferSize(0),
        m_pRing(NULL),
        m_pNotification(NULL),
        m_pvCompletionContext(NULL),
        m_FirstProcessing(TRUE),
        m_nNextBufferTime(0),
        m_nProcessed(0)
        {}

    ~CMiniportWaveRTStream();

    LONG                            m_RefCount;
    ULONG                           m_nStreamID;
    PVOID                           m_pPhysicalDevice;
    PPIN_DESCRIPTOR                 m_pPinDescriptor;
    WORD                            m_nSamplesPerFrame;
    WORD                            m_nChannels;
    ULONG                           m_nMaxNumberOfFrames;
    ULONG                           m_nBufferSize;          // bytes
    CPCMRing*                       m_pRing;
    PFN_WAVERT_PROCESS              m_pNotification;
    PVOID                           m_pvCompletionContext;
    BOOLEAN                         m_FirstProcessing;
    LONGLONG                        m_nNextBufferTime;      // 100 ns units
    volatile LONG                   m_nProcessed;           // frames delivered
    // Int16 staging copy of the PCM block for the DPC.
    INT16                           m_u64Samples;           // == 480*2
    INT16                           m_Samples[480*2];
};

typedef CMiniportWaveRTStream *PCMiniportWaveRTStream;
